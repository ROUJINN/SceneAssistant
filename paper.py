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
    parser.add_argument(
        "--json_path",
        type=str,
        default="outputs2/a_laundromat_section_with_two_front_loading_washin_20260228_183615/scene.json",
        help="Path to the scene.json file",
    )
    args = parser.parse_args()
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: File {json_path} not found.")
        sys.exit(1)
    with open(json_path, "r") as f:
        scene_state = json.load(f)
    json_dir = str(json_path.parent)
    scene_state = resolve_asset_paths(scene_state, json_dir)

    scene_state["camera"] = {
        "mode": "ViewScene",
        "args": {"view": "Front", "zoom": 1.5},
        "horizontal_rotation": 0.0,
        "vertical_rotation": 30.0,
        "movements": [],
    }

    # scene_state["camera"] = {
    #     "mode": "FocusOn",
    #     "args": {"view": "Iso", "target_name": "scorecard_holder", "zoom": 0.6},
    #     "horizontal_rotation": 0.0,
    #     "vertical_rotation": 0.0,
    #     "movements": [],
    # }
    status = blender_render.render_scene(
        scene_state,
        str(json_path.parent / "Front1.png"),
        render_mode=args.mode,
        # save_blend_path=str(json_path.parent / "scene.blend"),
    )
    collisions = status.get("collisions", [])
    if collisions:
        print(f"Warning: Collisions detected: {', '.join(collisions)}")
    else:
        print("No collisions detected.")


if __name__ == "__main__":
    main()
