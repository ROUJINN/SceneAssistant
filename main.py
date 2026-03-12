import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import blender_render
import bpy

# Local imports
import llm
import prompts

# Third party
import tyro
from image_gen import generate_image_with_zimage
from scene_manager import SceneManager
from threedgen import mesh_gen, rmbg


def create_wall_mesh(width, height, output_path):
    """Create a vertical wall mesh using Blender and save as GLB"""
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 1. Use a cube instead of a plane because walls are volumetric 3D objects.
    # size=1 creates a unit cube so scale values map directly to final dimensions.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    wall = bpy.context.active_object
    wall.name = "Wall"

    # 2. Set dimensions directly.
    # In GLB/Blender:
    # X = width, Y = thickness (0.05), Z = height.
    # With primitive_cube_add(size=1), scale values equal final dimensions.
    wall.scale = (width, 0.05, height)

    # 3. Optional position adjustment.
    # Cube center is at (0,0,0) by default; move Z by height/2 to place the base on ground.
    # wall.location.z = height / 2

    # 4. Apply transforms.
    # This bakes scale/rotation into mesh vertices and resets object scale to (1,1,1).
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Create white material
    mat = bpy.data.materials.new(name="WallMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # White color

    # Ensure the material is assigned.
    if wall.data.materials:
        wall.data.materials[0] = mat
    else:
        wall.data.materials.append(mat)

    # Export as GLB
    # Export with +Y Up (Blender default behavior).
    bpy.ops.export_scene.gltf(
        filepath=output_path, export_format="GLB", use_selection=True
    )

    return output_path


def parse_response(response_text):
    """
    Parses the LLM response for 'Action:'.
    Returns (actions_list)
    actions_list is a list of dicts: [{"type": "Func", "args": [...]}, ...]
    """
    actions = []

    # Extract Actions (JSON block)
    json_match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
    if json_match:
        try:
            json_content = json_match.group(1).strip()
            parsed = json.loads(json_content)
            if isinstance(parsed, list):
                actions = parsed
            else:
                # Handle case where it might be a single object instead of list
                actions = [parsed]
        except json.JSONDecodeError:
            print("Failed to parse JSON action block.")

    # Fallback: Try to find "Action: Func(...)" pattern if JSON failed or missing
    if not actions:
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\((.*)\)", response_text)
        if action_match:
            func_name = action_match.group(1)
            args_str = action_match.group(2)
            try:
                if args_str.strip().endswith(","):
                    args_str = args_str.strip()[:-1]
                # eval usage for legacy format support
                args = eval(f"[{args_str}]")
                actions.append({"type": func_name, "args": args})
            except Exception as e:
                print(f"Failed to parse fallback args: {args_str} | Error: {e}")

    return actions


def parse_reason(response_text):
    """
    Extracts the reasoning from the VLM response.
    Returns the reason string, or None if not found.
    """
    # Look for "Reason:" followed by text until "Action:" or end
    reason_match = re.search(
        r"Reason:\s*(.*?)(?=Action:|$)", response_text, re.DOTALL | re.IGNORECASE
    )
    if reason_match:
        return reason_match.group(1).strip()
    return None


class Agent:
    def __init__(
        self,
        goal: str,
        output_dir: str,
        model: str = "qwen",
        resume_step: int = None,
        human_editing_step: int = None,
    ):
        self.goal = goal
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.human_editing_step = human_editing_step

        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(exist_ok=True)

        self.scene = SceneManager(str(self.output_dir))

        # If resuming, trim chat history before initializing LLM
        if resume_step is not None:
            # Create temporary ChatRecorder to trim history
            temp_recorder = llm.ChatRecorder(
                run_id="agent_loop", base_dir=str(self.output_dir / "logs")
            )
            temp_recorder.trim_history_to_step(resume_step)

        self.llm = llm.LLM(
            model,
            run_id="agent_loop",
            log_dir=str(self.output_dir / "logs"),
            resume=resume_step is not None,
        )

        self.scene_file = self.output_dir / "scene.json"

        self.current_image_path = None
        self.step_count = 0
        self.system_message = None  # Store system message like collision warning

        # Resume logic
        if resume_step is not None:
            self._resume_from_step(resume_step)

    def _resume_from_step(self, step: int):
        """Resume scene state from a given step."""

        state_file = self.output_dir / f"step_{step:03d}_state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        # Restore scene state.
        self.scene.load_state(str(state_file))
        self.step_count = step - 1  # After loop increment, will become step

        # Set current image path.
        img_file = self.output_dir / f"step_{step:03d}.png"
        if img_file.exists():
            self.current_image_path = str(img_file)
        else:
            print(f"Warning: Image file not found: {img_file}")

        print(f"Resumed from step {step}")

    def _add_indoor_walls(self, length=6.0, width=6.0):
        """Add four walls for indoor scene context with custom dimensions

        Args:
            length: Room width along +X axis (meters)
            width: Room depth along -Y axis (meters)
        """
        wall_height = 3.0  # 3 meters high

        # Wall_Back at +Y, mesh_width = length (spans along X-axis)
        wall1_path = self.assets_dir / "wall_back.glb"
        create_wall_mesh(length, wall_height, str(wall1_path))
        self.scene.add_object(
            name="Wall_Back",
            description="white back wall",
            mesh_path=str(wall1_path),
            position=[0, width / 2, wall_height / 2],
        )

        # Wall_Left at -X, mesh_width = width (spans along Y-axis after rotation)
        wall2_path = self.assets_dir / "wall_left.glb"
        create_wall_mesh(width, wall_height, str(wall2_path))
        self.scene.add_object(
            name="Wall_Left",
            description="white left wall",
            mesh_path=str(wall2_path),
            position=[-length / 2, 0, wall_height / 2],
        )
        self.scene.get_object("Wall_Left")["rotation"] = [0, 0, 90]

        # Wall_Right at +X, mesh_width = width (spans along Y-axis after rotation)
        wall3_path = self.assets_dir / "wall_right.glb"
        create_wall_mesh(width, wall_height, str(wall3_path))
        self.scene.add_object(
            name="Wall_Right",
            description="white right wall",
            mesh_path=str(wall3_path),
            position=[length / 2, 0, wall_height / 2],
        )
        self.scene.get_object("Wall_Right")["rotation"] = [0, 0, 90]

        # Wall_Front at -Y, mesh_width = length (spans along X-axis)
        wall4_path = self.assets_dir / "wall_front.glb"
        create_wall_mesh(length, wall_height, str(wall4_path))
        self.scene.add_object(
            name="Wall_Front",
            description="white front wall",
            mesh_path=str(wall4_path),
            position=[0, -width / 2, wall_height / 2],
        )

        print(f"Added 4 walls: {length}m x {width}m room")

    def ensure_above_floor(self, name):
        obj = self.scene.get_object(name)
        if not obj:
            return

        bounds = self.scene.get_object_bounds(name)
        if bounds is None:
            return

        min_z = bounds[0][2]

        if min_z < -0.001:
            diff = -min_z
            obj["position"][2] += diff
            return f"Auto-corrected {name} Z-pos by +{diff:.3f} to be above floor."
        return None

    def render(self):
        print("Rendering scene...")

        state = {
            "objects": self.scene.objects,
            "camera": self.scene.camera_state,
            "floor_texture": self.scene.floor_texture,
        }

        img_name = f"step_{self.step_count:03d}.png"
        img_path = self.output_dir / img_name

        try:
            status = blender_render.render_scene(
                state, str(img_path), render_mode="agent"
            )

            self.current_image_path = str(img_path)

            collisions = status.get("collisions", [])
            if collisions:
                self.system_message = (
                    f"Warning: Collisions detected: {', '.join(collisions)}"
                )

            return True

        except Exception as e:
            print(f"Rendering Error: {e}")
            traceback.print_exc()
            return False

    def execute_action(self, func_name, args):
        print(f"Executing: {func_name}{args}")
        try:
            correction_msg = None

            if func_name == "Create":
                name, desc = args[0], args[1]
                if self.scene.object_exists(name):
                    return False, f"Object '{name}' already exists."

                print(f"Generating asset for {name}...")
                seed = int(time.time())
                raw_img = generate_image_with_zimage(desc, seed=seed)
                no_bg_img = rmbg(raw_img)
                mesh = mesh_gen(no_bg_img)
                glb_path = self.assets_dir / f"{name}.glb"
                mesh.export(str(glb_path))

                # Heuristic Placement
                max_z = 0.0
                for obj_name, obj_data in self.scene.objects.items():
                    bounds = self.scene.get_object_bounds(obj_name)
                    if bounds is not None and bounds[1][2] > max_z:
                        max_z = bounds[1][2]

                self.scene.add_object(
                    name, desc, str(glb_path), position=[0.0, 0.0, 0.0]
                )

                obj_bounds = self.scene.get_object_bounds(name)
                min_z_local = obj_bounds[0][2]

                shift = max_z - min_z_local
                self.scene.get_object(name)["position"][2] = shift

                correction_msg = self.ensure_above_floor(name)

                msg = f"Created {name}"
                if correction_msg:
                    msg += f" ({correction_msg})"
                return True, msg

            elif func_name == "Translate":
                name, axis, val = args
                obj = self.scene.get_object(name)
                if not obj:
                    return False, f"Object {name} not found."

                idx = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
                obj["position"][idx] += float(val)

                correction_msg = self.ensure_above_floor(name)
                msg = f"Translated {name} {axis} by {val}"
                if correction_msg:
                    msg += f" {correction_msg}"
                return True, msg

            elif func_name == "Place":
                name, pos = args
                obj = self.scene.get_object(name)
                if not obj:
                    return False, f"Object {name} not found."

                obj["position"] = [float(x) for x in pos]

                correction_msg = self.ensure_above_floor(name)
                msg = f"Placed {name} at {pos}"
                if correction_msg:
                    msg += f" {correction_msg}"
                return True, msg

            elif func_name == "Rotate":
                name, axis, val = args
                obj = self.scene.get_object(name)
                if not obj:
                    return False, f"Object {name} not found."

                idx = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
                obj["rotation"][idx] = float(val)

                correction_msg = self.ensure_above_floor(name)
                msg = f"Rotated {name} around {axis} to {val} degrees."
                if correction_msg:
                    msg += f" {correction_msg}"
                return True, msg

            elif func_name == "Scale":
                name, val = args
                obj = self.scene.get_object(name)
                if not obj:
                    return False, f"Object {name} not found."

                # Handle scalar vs list input
                if isinstance(val, (int, float)):
                    scale_vec = [float(val), float(val), float(val)]
                elif isinstance(val, (list, tuple)) and len(val) == 3:
                    scale_vec = [float(v) for v in val]
                else:
                    return (
                        False,
                        f"Invalid scale value: {val}. Must be number or [x, y, z].",
                    )

                # Apply scaling (ABSOLUTE in terms of scale factors)
                obj["scale"] = scale_vec

                correction_msg = self.ensure_above_floor(name)
                msg = f"Scaled {name} to {val}"
                if correction_msg:
                    msg += f" {correction_msg}"
                return True, msg

            elif func_name == "Duplicate":
                name, count = args
                created_names = self.scene.duplicate_object(name, int(count))
                if not created_names:
                    return False, f"Failed to duplicate {name}"

                msgs = []
                for new_name in created_names:
                    # Heuristic Placement: Place on top of the highest object (same as Create)
                    max_z = 0.0
                    for obj_name in self.scene.objects:
                        # Skip self to avoid self-reference issues if bounds overlap (though duplications shouldn't yet)
                        if obj_name == new_name:
                            continue
                        bounds = self.scene.get_object_bounds(obj_name)
                        if bounds is not None and bounds[1][2] > max_z:
                            max_z = bounds[1][2]

                    # Reset position to (0,0,0) before calculating Z, just like Create starts at center
                    self.scene.get_object(new_name)["position"] = [0.0, 0.0, 0.0]

                    obj_bounds = self.scene.get_object_bounds(new_name)
                    if obj_bounds is not None:
                        min_z_local = obj_bounds[0][2]
                        shift = max_z - min_z_local
                        self.scene.get_object(new_name)["position"][2] = shift

                    c = self.ensure_above_floor(new_name)
                    if c:
                        msgs.append(c)

                msg = f"Duplicated {name} into {created_names} with Heuristic Placement"
                if msgs:
                    msg += " " + " ".join(msgs)
                return True, msg

            elif func_name == "Delete":
                name = args[0]
                if self.scene.delete_object(name):
                    return True, f"Deleted {name}"
                else:
                    return False, f"Object {name} not found."

            elif func_name == "ViewScene":
                view = args[0] if len(args) > 0 else "Iso"
                zoom = float(args[1]) if len(args) > 1 else 1.0
                self.scene.set_camera("ViewScene", view=view, zoom=zoom)
                return True, f"Camera set to ViewScene(view='{view}', zoom={zoom})"

            elif func_name == "FocusOn":
                target_name = args[0]
                view = args[1] if len(args) > 1 else "Iso"
                zoom = float(args[2]) if len(args) > 2 else 1.0

                if not self.scene.object_exists(target_name):
                    return False, f"Object '{target_name}' not found."

                self.scene.set_camera(
                    "FocusOn", target_name=target_name, view=view, zoom=zoom
                )
                return (
                    True,
                    f"Camera focused on {target_name} (view='{view}', zoom={zoom})",
                )

            elif func_name == "RotateCamera":
                horizontal = float(args[0])
                vertical = float(args[1])
                self.scene.rotate_camera(horizontal, vertical)
                return (
                    True,
                    f"Camera rotated by horizontal={horizontal}°, vertical={vertical}°",
                )

            elif func_name == "MoveCamera":
                direction = args[0]
                distance = float(args[1])
                self.scene.move_camera(direction, distance)
                return True, f"Camera moved {direction} by {distance}"

            elif func_name == "GenerateFloorTexture":
                description = args[0]
                print(f"Generating floor texture: {description}...")

                seed = int(time.time())
                texture_img = generate_image_with_zimage(description, seed=seed)

                texture_filename = f"floor_texture_{seed}.png"
                texture_path = self.assets_dir / texture_filename
                texture_img.save(str(texture_path))

                self.scene.floor_texture = str(texture_path)
                return True, f"Generated floor texture: {description}"

            elif func_name == "AddIndoorWalls":
                room_length = float(args[0]) if len(args) > 0 else 6.0
                room_width = float(args[1]) if len(args) > 1 else 6.0

                # Delete existing walls
                wall_names = ["Wall_Back", "Wall_Left", "Wall_Right", "Wall_Front"]
                removed_walls = []
                for wall_name in wall_names:
                    if self.scene.object_exists(wall_name):
                        self.scene.delete_object(wall_name)
                        removed_walls.append(wall_name)

                # Create new walls
                try:
                    self._add_indoor_walls(room_length, room_width)
                    msg = f"Added indoor walls: {room_length}m x {room_width}m room"
                    if removed_walls:
                        msg += f" (replaced: {', '.join(removed_walls)})"
                    return True, msg
                except Exception as e:
                    return False, f"Failed to add indoor walls: {str(e)}"

            elif func_name == "Finish":
                self.scene.save_state(str(self.scene_file))
                print(f"Scene state saved to {self.scene_file}")
                return True, "Finished"

            else:
                return False, f"Unknown action: {func_name}"

        except Exception as e:
            traceback.print_exc()
            return False, f"Error executing {func_name}: {str(e)}"

    def validate_action_batch(self, actions):
        """
        Validate that Create and Duplicate actions are not mixed with other actions.

        Returns:
            (valid: bool, message: str)
        """
        create_duplicate_actions = {"Create", "Duplicate"}
        modify_actions = {"Rotate", "Place", "Scale", "Translate"}

        has_create_or_duplicate = any(
            action.get("type") in create_duplicate_actions for action in actions
        )
        has_modify_action = any(
            action.get("type") in modify_actions for action in actions
        )

        if has_create_or_duplicate and has_modify_action:
            return (
                False,
                "Invalid action batch: Create and Duplicate actions must be executed in a separate batch. Do not mix Create/Duplicate with other actions (e.g., Rotate, Place, Scale, etc.) in the same batch.",
            )

        return True, None

    def get_action_history_with_system(self) -> str:
        """
        Reconstruct action history from LLM's chat_history.json.
        Each action is paired with the system message from the user prompt that preceded it.

        The history format is: [system, user, assistant, system, user, assistant, ...]
        For each assistant message at index i, the system message is in the user message at index i-1.

        Returns:
            A string with each action formatted as "[System: ...] Step N: Action(...)"
            or "Step N: Action(...)" if no system message.
        """
        history = self.llm.recorder.load_history()

        action_history = []
        step_counter = 0  # Track actual step count (excluding verification messages)

        i = 0
        while i < len(history):
            if i + 2 >= len(history):
                break

            # Get system message from the user message before this assistant
            user_msg = history[i + 1]
            system_message = "None"
            if user_msg.get("role") == "user" and user_msg.get("content"):
                for item in user_msg["content"]:
                    if item.get("type") == "text":
                        user_text = item.get("text", "")
                        # Extract "# System Messages" section
                        sys_pattern = r"# System Messages\s*\n(.+?)(?=\n#|$)"
                        sys_match = re.search(sys_pattern, user_text, re.DOTALL)
                        if sys_match:
                            system_message = sys_match.group(1).strip()
                        break

            # Get assistant message (contains actions)
            assistant_msg = history[i + 2]
            assistant_text = ""
            if assistant_msg.get("role") == "assistant":
                assistant_text = assistant_msg.get("content", "")

            # Parse actions from assistant's JSON response
            actions = []
            code_pattern = r"```json(.*?)```"
            match = re.search(code_pattern, assistant_text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                try:
                    action_list = json.loads(code)
                    for action in action_list:
                        if not isinstance(action, dict):
                            continue
                        func_name = action.get("type", "")
                        args = action.get("args", [])
                        args_str = ", ".join(
                            [
                                f"'{arg}'" if isinstance(arg, str) else str(arg)
                                for arg in args
                            ]
                        )
                        actions.append(f"{func_name}({args_str})")
                except json.JSONDecodeError:
                    pass

            # Only add to action_history if actions were found (skip verification messages)
            if actions:
                step_counter += 1
                for idx, action in enumerate(actions):
                    if idx == 0 and system_message and system_message != "None":
                        action_history.append(
                            f"[System: {system_message}]\nStep {step_counter}: {action}"
                        )
                    else:
                        action_history.append(f"Step {step_counter}: {action}")

            i += 3

        if not action_history:
            return "No actions taken yet."

        return "\n".join(action_history)

    def _get_human_feedback(self) -> str:
        """Get feedback from the human editor.

        Returns:
            str: Feedback text entered by the user.
        """
        img_file = self.output_dir / f"step_{self.step_count:03d}.png"
        print(f"\n=== Human Editing - Step {self.step_count} ===")
        print(f"Please check the scene image: {img_file}")
        print(
            "Enter your feedback/instruction for this step, or 'q' to finish and save the scene:"
        )
        try:
            user_input = input("Your feedback: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = "q"
        return user_input

    def run(self):
        print("Starting Agent Loop...")

        max_steps = 20
        # In human-edit mode, there is no max-step limit.
        while self.step_count < max_steps or (
            self.human_editing_step is not None
            and self.step_count >= self.human_editing_step
        ):
            self.step_count += 1
            print(f"\n--- Step {self.step_count} ---")

            # Save current state (state BEFORE executing actions for this step)
            # step_i.png = state before step i execution
            # step_i_state.json = state before step i execution
            if not self.render():
                print(f"Render failed at step {self.step_count}. Exiting.")
                return

            state_path = self.output_dir / f"step_{self.step_count:03d}_state.json"
            self.scene.save_state(str(state_path))

            # Human Editing Mode: Get human feedback if enabled
            if (
                self.human_editing_step is not None
                and self.step_count >= self.human_editing_step
            ):
                feedback = self._get_human_feedback()
                if feedback.lower() == "q":
                    print("Human approved. Saving final scene state...")
                    # Save final state
                    self.scene.save_state(str(self.scene_file))
                    print(f"Scene state saved to {self.scene_file}")
                    print("Goal Achieved!")
                    return
                else:
                    # Feed human feedback into the next prompt as a system message.
                    self.system_message = f"Human feedback: {feedback}"

            system_messages_str = self.system_message if self.system_message else "None"

            user_prompt_str = prompts.USER_PROMPT_TEMPLATE.format(
                user_instruction=self.goal,
                action_summary=self.get_action_history_with_system(),
                scene_json=self.scene.get_scene_json(),
                system_messages=system_messages_str,
            )

            print("Thinking...")

            # Clear system message after sending it
            self.system_message = None
            images = []
            if self.current_image_path:
                images.append(self.current_image_path)

            response = self.llm.chat(
                user_text=user_prompt_str,
                image_path=images,
                system_text=prompts.SYSTEM_PROMPT,
            )

            print(f"Agent says: {response}")

            # Extract and print reasoning
            reason = parse_reason(response)
            if reason:
                print(f"Reasoning: {reason}")

            actions = parse_response(response)

            if not actions:
                print("No valid actions found.")
                continue

            # Validate action batch - Create/Duplicate should not be mixed with other actions
            is_valid, error_msg = self.validate_action_batch(actions)
            if not is_valid:
                print(f"Action validation failed: {error_msg}")
                # Set system message to warn the agent for the next step
                self.system_message = f"Action batch validation failed: {error_msg}"
                # Skip execution and continue to next step
                continue

            # Batch Execution Loop
            batch_success = True
            actions_executed = 0

            for action in actions:
                func_name = action.get("type")
                args = action.get("args", [])

                success, msg = self.execute_action(func_name, args)

                if success:
                    actions_executed += 1

                if not success:
                    print(f"Action Failed: {msg}")
                    batch_success = False
                    break  # Stop executing subsequent actions in this batch if one fails

            if not batch_success:
                # Rollback to state before this step
                state_path = self.output_dir / f"step_{self.step_count:03d}_state.json"
                if state_path.exists():
                    self.scene.load_state(str(state_path))
                    print(f"Rolled back scene to state before step {self.step_count}")
                else:
                    print(f"Warning: State file not found for rollback: {state_path}")

                # Rollback LLM chat history to before this step
                self.llm.recorder.trim_history_to_step(self.step_count)
                self.llm.reload_history()  # Reload trimmed history into memory
                print(f"Rolled back LLM chat history to before step {self.step_count}")

                print(f"Step {self.step_count} failed. Repeating step...")
                self.step_count -= 1  # Decrement step count to repeat the current step
                continue

            # Check if finished
            if batch_success and actions and actions[-1].get("type") == "Finish":
                print("Agent requested finish.")
                self.scene.save_state(str(self.scene_file))
                print(f"Scene state saved to {self.scene_file}")
                print("Goal Achieved!")
                break

        # Save final scene if we reached max steps without finishing
        if self.step_count >= max_steps:
            # Increment step to represent the state AFTER the last action
            self.step_count += 1
            print(f"\n--- Step {self.step_count} (Final State) ---")

            # Render and save the final state
            if self.render():
                state_path = self.output_dir / f"step_{self.step_count:03d}_state.json"
                self.scene.save_state(str(state_path))
                print(f"Final state saved to {state_path}")

            # Also save to scene.json for convenience
            self.scene.save_state(str(self.scene_file))
            print(
                f"Reached max steps ({max_steps}). Scene state also saved to {self.scene_file}"
            )


@dataclass
class Args:
    prompt: str = "A stack of four books"
    output_dir: str = "outputs/agent_session"
    model: str = "qwen"
    resume_step: Optional[int] = None
    human_editing_step: Optional[int] = (
        None  # Human-edit mode: from this step onward, each step requires human feedback.
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    agent = Agent(
        args.prompt,
        args.output_dir,
        args.model,
        args.resume_step,
        args.human_editing_step,
    )
    agent.run()
