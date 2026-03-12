import copy
import json
import os
from typing import Any, Dict, List

import bpy
import numpy as np
from mathutils import Matrix, Vector
from scipy.spatial.transform import Rotation as R


class SceneManager:
    def __init__(self, output_dir: str = None):
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.camera_state = {
            "mode": "ViewScene",
            "args": {"view": "Iso", "zoom": 1.0},
            "horizontal_rotation": 0.0,
            "vertical_rotation": 0.0,
            "movements": [],
        }
        self.floor_texture: str = None  # Path to floor texture image
        self.output_dir = output_dir  # Base directory for relative path conversion

    def _compute_local_bounds(self, mesh_path: str):
        """
        Computes the local axis-aligned bounding box of the mesh at mesh_path.
        Uses bpy to import and measure.
        Returns (min_vec, max_vec) as numpy arrays.
        """
        # We need to run this in a way that doesn't mess up if there's an ongoing scene context,
        # but since this is called sequentially in the agent loop, it's generally safe to clear the scene.
        # Ensure we are in object mode
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.wm.read_factory_settings(use_empty=True)

        if not os.path.exists(mesh_path):
            return np.zeros(3), np.zeros(3)

        # Suppress output if possible, but hard in python script inside blender
        try:
            if mesh_path.endswith(".glb") or mesh_path.endswith(".gltf"):
                bpy.ops.import_scene.gltf(filepath=mesh_path)
            elif mesh_path.endswith(".obj"):
                bpy.ops.wm.obj_import(filepath=mesh_path)
            else:
                print(f"Unsupported mesh format: {mesh_path}")
                return np.zeros(3), np.zeros(3)
        except Exception as e:
            print(f"Failed to import {mesh_path}: {e}")
            return np.zeros(3), np.zeros(3)

        min_b = Vector((float("inf"), float("inf"), float("inf")))
        max_b = Vector((float("-inf"), float("-inf"), float("-inf")))
        found_mesh = False

        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                found_mesh = True
                # Calculate world coords of vertices (relative to import root at 0,0,0)
                # This gives us the "Model Space" bounds.
                bbox = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
                for v in bbox:
                    min_b.x = min(min_b.x, v.x)
                    min_b.y = min(min_b.y, v.y)
                    min_b.z = min(min_b.z, v.z)
                    max_b.x = max(max_b.x, v.x)
                    max_b.y = max(max_b.y, v.y)
                    max_b.z = max(max_b.z, v.z)

        if not found_mesh:
            return np.zeros(3), np.zeros(3)

        return np.array(min_b), np.array(max_b)

    def add_object(self, name: str, description: str, mesh_path: str, position=None):
        if position is None:
            position = [0.0, 0.0, 0.0]

        # Compute and cache local bounds
        min_b, max_b = self._compute_local_bounds(mesh_path)

        self.objects[name] = {
            "name": name,
            "description": description,
            "mesh_path": mesh_path,
            "position": position,
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "local_bounds": (min_b.tolist(), max_b.tolist()),
        }

    def duplicate_object(self, name: str, count: int):
        if name not in self.objects:
            return []

        created_names = []
        original = self.objects[name]

        # Find the highest existing copy number
        base_prefix = f"{name}_copy_"
        max_copy_num = 0

        for obj_name in self.objects:
            if obj_name.startswith(base_prefix):
                # Extract the number after base_prefix
                suffix = obj_name[len(base_prefix) :]
                if suffix.isdigit():
                    num = int(suffix)
                    max_copy_num = max(max_copy_num, num)

        # Start from the next available number
        next_num = max_copy_num + 1

        for i in range(count):
            new_name = f"{name}_copy_{next_num}"
            next_num += 1

            new_obj = copy.deepcopy(original)
            new_obj["name"] = new_name
            # Do not offset here; let the agent handle placement logic (e.g. stacking)

            self.objects[new_name] = new_obj
            created_names.append(new_name)

        return created_names

    def get_object(self, name: str):
        return self.objects.get(name)

    def delete_object(self, name: str):
        if name in self.objects:
            del self.objects[name]
            return True
        return False

    def object_exists(self, name: str):
        return name in self.objects

    def update_transform(self, name: str, transform_type: str, new_value):
        if name in self.objects:
            self.objects[name][transform_type] = new_value
            return True
        return False

    def set_camera(self, mode: str, **kwargs):
        # Reset relative transformations when setting a new base camera mode
        self.camera_state = {
            "mode": mode,
            "args": kwargs,
            "horizontal_rotation": 0.0,
            "vertical_rotation": 0.0,
            "movements": [],
        }

    def rotate_camera(self, horizontal: float, vertical: float):
        """Rotate camera relative to current state"""
        # Accumulate rotations on top of current base view
        h_rot = self.camera_state.get("horizontal_rotation", 0.0)
        v_rot = self.camera_state.get("vertical_rotation", 0.0)

        self.camera_state["horizontal_rotation"] = h_rot + horizontal
        self.camera_state["vertical_rotation"] = v_rot + vertical

    def move_camera(self, direction: str, distance: float):
        """Move camera relative to current state"""
        # Accumulate movements on top of current base view
        movements = self.camera_state.get("movements", []).copy()
        movements.append({"direction": direction, "distance": distance})
        self.camera_state["movements"] = movements

    def get_object_bounds(self, name):
        """
        Returns the World Space Axis-Aligned Bounding Box (AABB)
        as a numpy array [[min_x, min_y, min_z], [max_x, max_y, max_z]].
        """
        obj = self.objects.get(name)
        if not obj:
            return None

        local_min = np.array(obj["local_bounds"][0])
        local_max = np.array(obj["local_bounds"][1])

        # Construct the 8 corners of the local bounding box
        corners = []
        for x in [local_min[0], local_max[0]]:
            for y in [local_min[1], local_max[1]]:
                for z in [local_min[2], local_max[2]]:
                    corners.append([x, y, z])
        corners = np.array(corners)

        # Apply transformations
        pos = obj["position"]
        rot = obj["rotation"]
        scale = obj["scale"]

        # 1. Scale
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = scale

        # 2. Rotate
        r = R.from_euler("xyz", rot, degrees=True)
        M_rot = np.eye(4)
        M_rot[:3, :3] = r.as_matrix()

        # 3. Translate
        T = np.eye(4)
        T[:3, 3] = pos

        # Combined Matrix
        M = T @ M_rot @ S

        # Transform corners
        # Extend corners to 4D homogeneous coordinates
        ones = np.ones((8, 1))
        corners_4d = np.hstack([corners, ones])

        transformed_corners = (M @ corners_4d.T).T[:, :3]

        min_bounds = transformed_corners.min(axis=0)
        max_bounds = transformed_corners.max(axis=0)

        return np.array([min_bounds, max_bounds])

    def get_scene_json(self):
        simplified_objects = []
        for name, obj in self.objects.items():
            # Use cached local bounds for "size" and apply current scale
            local_min = np.array(obj["local_bounds"][0])
            local_max = np.array(obj["local_bounds"][1])
            local_dims = local_max - local_min

            # Apply absolute scale to dimensions
            scale = np.array(obj["scale"])
            scaled_dims = local_dims * np.abs(scale)
            dimensions = [round(x, 2) for x in scaled_dims]

            simplified_objects.append(
                {
                    "name": name,
                    "pos": [round(x, 2) for x in obj["position"]],
                    "rot": [round(x, 2) for x in obj["rotation"]],
                    "scale": [round(x, 2) for x in obj["scale"]],
                    "size": dimensions,  # [width(x), depth(y), height(z)] - Scaled Local Size
                }
            )
        return json.dumps(simplified_objects, indent=2)

    def save_state(self, path: str):
        # Convert absolute paths to relative paths before saving
        objects_copy = {}
        for name, obj_data in self.objects.items():
            obj_copy = obj_data.copy()
            if "mesh_path" in obj_copy and self.output_dir:
                abs_path = os.path.abspath(obj_copy["mesh_path"])
                try:
                    rel_path = os.path.relpath(abs_path, self.output_dir)
                    obj_copy["mesh_path"] = rel_path
                except ValueError:
                    # Fallback to absolute path if on different drives
                    pass
            objects_copy[name] = obj_copy

        floor_texture_rel = self.floor_texture
        if self.floor_texture and self.output_dir:
            abs_path = os.path.abspath(self.floor_texture)
            try:
                floor_texture_rel = os.path.relpath(abs_path, self.output_dir)
            except ValueError:
                # Fallback to absolute path if on different drives
                pass

        state = {
            "objects": objects_copy,
            "camera": self.camera_state,
            "floor_texture": floor_texture_rel,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            state = json.load(f)

        # Convert relative paths to absolute paths when loading
        objects_state = state.get("objects", {})
        for name, obj_data in objects_state.items():
            if "mesh_path" in obj_data and self.output_dir:
                mesh_path = obj_data["mesh_path"]
                if not os.path.isabs(mesh_path):
                    # Relative path - convert to absolute
                    abs_path = os.path.abspath(os.path.join(self.output_dir, mesh_path))
                    obj_data["mesh_path"] = abs_path

        self.objects = objects_state
        self.camera_state = state.get(
            "camera",
            {
                "mode": "ViewScene",
                "args": {"view": "Iso", "zoom": 1.0},
                "horizontal_rotation": 0.0,
                "vertical_rotation": 0.0,
                "movements": [],
            },
        )

        floor_texture = state.get("floor_texture", None)
        if floor_texture and self.output_dir and not os.path.isabs(floor_texture):
            # Relative path - convert to absolute
            floor_texture = os.path.abspath(
                os.path.join(self.output_dir, floor_texture)
            )
        self.floor_texture = floor_texture
        return True
