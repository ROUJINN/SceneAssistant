import argparse
import json
import os
import sys
from pathlib import Path

import blender_render


def resolve_asset_paths(state, base_dir):
    """Convert relative asset paths to absolute paths based on base_dir."""
    # Resolve mesh paths in objects
    objects = state.get("objects", {})
    for name, obj_data in objects.items():
        if "mesh_path" in obj_data:
            mesh_path = obj_data["mesh_path"]
            if not os.path.isabs(mesh_path):
                # Convert relative path to absolute
                abs_path = os.path.abspath(os.path.join(base_dir, mesh_path))
                obj_data["mesh_path"] = abs_path

    # Resolve floor texture path if present
    if "floor_texture" in state and state["floor_texture"]:
        texture_path = state["floor_texture"]
        if not os.path.isabs(texture_path):
            abs_path = os.path.abspath(os.path.join(base_dir, texture_path))
            state["floor_texture"] = abs_path

    return state


def main():
    parser = argparse.ArgumentParser(
        description="Render a scene from a scene.json file."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="outputs2/a_japanese_street_corner_scene_with_a_two_story_de_20260222_014759/step_012_state.json",
        help="Path to the scene.json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: same dir as json)",
    )
    parser.add_argument(
        "--save_blend",
        action="store_true",
        help="If set, saves the Blender scene (.blend) alongside the render.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="final",
        choices=["agent", "final"],
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: File {json_path} not found.")
        sys.exit(1)

    with open(json_path, "r") as f:
        scene_state = json.load(f)

    # Resolve relative asset paths to absolute paths based on JSON file's directory
    json_dir = str(json_path.parent)
    scene_state = resolve_asset_paths(scene_state, json_dir)

    if args.output:
        output_path = args.output
    else:
        # Default to scene_render.png in the same folder
        output_path = str(json_path.parent / "scene_final_render.png")

    save_blend_path = None
    if args.save_blend:
        # Derive .blend path from output_path
        save_blend_path = str(Path(output_path).with_suffix(".blend"))

    print(f"Loading scene from {json_path}...")
    print(f"Rendering to {output_path}...")
    if save_blend_path:
        print(f"Saving blend file to {save_blend_path}...")

    # overwrite camera settings
    scene_state["camera"] = {
        "mode": "ViewScene",
        "args": {"view": "Iso", "zoom": 1.3},
        "horizontal_rotation": 0.0,
        "vertical_rotation": 0.0,
        "movements": [],
    }

    # scene_state["camera"] = {
    #     "mode": "FocusOn",
    #     "args": {"target_name": "campfire_ring", "view": "Iso", "zoom": 1.5},
    #     "horizontal_rotation": 0.0,
    #     "vertical_rotation": 0.0,
    #     "movements": [],
    # }
    # scene_state["floor_texture"] = "/home/lj/3D/code-scene-gen/images/grass_texture.png"

    blender_render.render_scene(
        scene_state,
        output_path,
        save_blend_path=save_blend_path,
        render_mode=args.mode,
    )
    print("Render complete!")


if __name__ == "__main__":
    main()
