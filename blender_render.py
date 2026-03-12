import contextlib
import math
import os
import sys

import bpy
import bpy_extras
import cv2
import numpy as np
from icecream import ic
from mathutils import Euler, Matrix, Vector

ic.disable()


@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    """
    Redirect system-level stdout and stderr.
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # Explicitly close Python stdout.
        os.dup2(to.fileno(), fd)  # Redirect system fd to target file.
        sys.stdout = os.fdopen(fd, "w")  # Reopen Python stdout.

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # Execute wrapped code here.
        finally:
            _redirect_stdout(to=old_stdout)  # Restore original output.


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_world(hdri_path=None, hdri_strength=1.0):
    """Setup world environment with optional HDRI.

    Args:
        hdri_path: Path to HDRI image file (.hdr or .exr). If None, uses simple background.
        hdri_strength: Strength multiplier for HDRI lighting (default: 1.0)
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    # Enable world nodes
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # Create output node
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (300, 0)

    if hdri_path and os.path.exists(hdri_path):
        # HDRI Environment
        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-800, 0)

        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.location = (-600, 0)

        env_texture = nodes.new(type="ShaderNodeTexEnvironment")
        env_texture.location = (-300, 0)
        try:
            env_texture.image = bpy.data.images.load(hdri_path)
        except Exception as e:
            print(f"Warning: Failed to load HDRI {hdri_path}: {e}")
            # Fallback to simple background
            bg_node = nodes.new(type="ShaderNodeBackground")
            bg_node.location = (0, 0)
            bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1)
            bg_node.inputs["Strength"].default_value = 1.0
            links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
            return

        bg_node = nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Strength"].default_value = hdri_strength

        # Connect nodes
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env_texture.inputs["Vector"])
        links.new(env_texture.outputs["Color"], bg_node.inputs["Color"])
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
    else:
        # Simple background when no HDRI
        bg_node = nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1)
        bg_node.inputs["Strength"].default_value = 1.0
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def get_objects_bounds(object_names):
    """Calculate the XY bounds of all objects for floor texture coverage."""
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    # Update scene to ensure transforms are applied
    bpy.context.view_layer.update()

    for name in object_names:
        obj = bpy.data.objects.get(name)
        if not obj:
            continue

        # Include all child meshes in the hierarchy
        objects_to_check = [obj]
        objects_to_check.extend(
            [child for child in obj.children_recursive if child.type == "MESH"]
        )

        for check_obj in objects_to_check:
            if check_obj.type != "MESH":
                continue

            # Get world-space bounding box
            bbox_world = [
                check_obj.matrix_world @ Vector(v) for v in check_obj.bound_box
            ]
            for v in bbox_world:
                min_x = min(min_x, v.x)
                max_x = max(max_x, v.x)
                min_y = min(min_y, v.y)
                max_y = max(max_y, v.y)

    # No margin - cover exactly to object bounds
    margin = 1
    if min_x == float("inf"):
        # No objects found, return default range
        return -5.0, 5.0, -5.0, 5.0

    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def create_floor_texture(texture_path, min_x, max_x, min_y, max_y):
    # 1. Compute physical dimensions.
    size_x = max_x - min_x
    size_y = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 2. Create the plane.
    bpy.ops.mesh.primitive_plane_add(size=2, location=(center_x, center_y, 0))
    plane = bpy.context.active_object
    plane.scale = (size_x / 2, size_y / 2, 1.0)

    # Apply scale so UV mapping spans 0..1 over the full object.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 3. Configure material nodes.
    mat = bpy.data.materials.new(name="FloorTextureMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create nodes (TexCoord -> Mapping -> TexImage -> BSDF).
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    tex_image = nodes.new(type="ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(texture_path)
    tex_image.extension = "REPEAT"

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf.inputs["Roughness"].default_value = 0.95  # Increase roughness to reduce glare.
    bsdf.inputs["Specular IOR Level"].default_value = 0.01  # Specular IOR level.
    # --- Key improvement: distortion-free tiling control ---

    # Decide the real-world size represented by this 1024x1024 texture.
    # Example: make it cover exactly 1m x 1m in Blender.
    unit_size = 1

    # Set Mapping scale to physical_size / unit_size.
    # Then X tiles size_x times and Y tiles size_y times.
    # Matching X/Y physical ratios prevents texture stretching.
    mapping.inputs["Scale"].default_value = (
        size_x / unit_size,
        size_y / unit_size,
        1.0,
    )

    # Connect nodes.
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
    links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    plane.data.materials.append(mat)
    return plane


def clamp_position(x, y, w, h, padding=15):
    """Clamp a position to stay within image bounds with padding."""
    x = max(padding, min(x, w - padding))
    y = max(padding, min(y, h - padding))
    return x, y


def arrange_labels_left_column(labels, w, h):
    """Arrange all labels in a vertical column on the left side.

    Returns list of dicts with:
        - name: label text
        - anchor_x, anchor_y: original object position
        - text_x, text_y: adjusted text position (bottom-left corner)
        - text_w, text_h: text dimensions
    """
    label_rects = []

    # Calculate dimensions for all labels
    for label in labels:
        name = label["name"]
        x = label["x"]
        y = label["y"]

        px = int(x * w)
        py = int((1.0 - y) * h)

        (text_w, text_h), baseline = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        label_rects.append(
            {
                "name": name,
                "anchor_x": px,
                "anchor_y": py,
                "text_w": text_w,
                "text_h": text_h,
            }
        )

    # Arrange labels in a left column
    left_margin = 15
    top_margin = 30
    line_spacing = 25

    for i, rect in enumerate(label_rects):
        rect["text_x"] = left_margin
        rect["text_y"] = top_margin + i * line_spacing

    return label_rects


def draw_hud_on_image(image_path, cam_matrix, labels=[]):
    """Draw HUD with coordinate axes and object labels on the rendered image (transparent)."""
    if not os.path.exists(image_path):
        return

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return

    h, w = img.shape[:2]

    # --- HUD Coordinate Axes ---
    center_x = w - 100
    center_y = 100
    axis_length = 75
    thickness = 4

    cam_rot = np.array(cam_matrix)[:3, :3]
    world_to_cam = cam_rot.T

    axes = {
        "X": np.array([1, 0, 0]),
        "Y": np.array([0, 1, 0]),
        "Z": np.array([0, 0, 1]),
    }
    colors = {"X": (0, 0, 255), "Y": (0, 255, 0), "Z": (255, 0, 0)}  # BGR
    sorted_axes = ["X", "Y", "Z"]

    # For BGRA images, separate RGB and Alpha, draw on RGB, then recombine
    has_alpha = img.shape[2] == 4
    if has_alpha:
        alpha_channel = img[:, :, 3].copy()
        img_rgb = img[:, :, :3].copy()
        alpha_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        img_rgb = img
        alpha_mask = None

    # Draw center point
    cv2.circle(img_rgb, (center_x, center_y), 4, (200, 200, 200), -1)
    if alpha_mask is not None:
        cv2.circle(alpha_mask, (center_x, center_y), 4, 255, -1)

    for axis_name in sorted_axes:
        vec_world = axes[axis_name]
        vec_cam = np.dot(world_to_cam, vec_world)
        x_screen = vec_cam[0]
        y_screen = vec_cam[1]

        end_x = int(center_x + x_screen * axis_length)
        end_y = int(center_y - y_screen * axis_length)

        # Draw axis line
        cv2.line(
            img_rgb,
            (center_x, center_y),
            (end_x, end_y),
            colors[axis_name],
            thickness,
            lineType=cv2.LINE_AA,
        )
        if alpha_mask is not None:
            cv2.line(
                alpha_mask,
                (center_x, center_y),
                (end_x, end_y),
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )

        # Draw axis label (clamp position to stay within image bounds)
        label_x = int(center_x + x_screen * (axis_length + 20))
        label_y = int(center_y - y_screen * (axis_length + 20))
        label_x, label_y = clamp_position(label_x, label_y, w, h, padding=15)
        cv2.putText(
            img_rgb,
            axis_name,
            (label_x - 5, label_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colors[axis_name],
            2,
            cv2.LINE_AA,
        )
        if alpha_mask is not None:
            cv2.putText(
                alpha_mask,
                axis_name,
                (label_x - 5, label_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                255,
                2,
                cv2.LINE_AA,
            )

    # --- Object Labels in Left Column ---
    if labels:
        arranged_labels = arrange_labels_left_column(labels, w, h)

        for label_rect in arranged_labels:
            name = label_rect["name"]
            anchor_x = label_rect["anchor_x"]
            anchor_y = label_rect["anchor_y"]
            text_x = int(label_rect["text_x"])
            text_y = int(label_rect["text_y"])
            text_w = label_rect["text_w"]
            text_h = label_rect["text_h"]

            # Calculate label center for leader line endpoint
            label_right_x = text_x + text_w + 5
            label_center_y = text_y - text_h // 2

            # Always draw leader line from object to label
            cv2.line(
                img_rgb,
                (anchor_x, anchor_y),
                (label_right_x, label_center_y),
                (120, 120, 120),
                1,
                lineType=cv2.LINE_AA,
            )
            if alpha_mask is not None:
                cv2.line(
                    alpha_mask,
                    (anchor_x, anchor_y),
                    (label_right_x, label_center_y),
                    255,
                    1,
                    lineType=cv2.LINE_AA,
                )

            # Draw anchor point (small circle at object location)
            cv2.circle(img_rgb, (anchor_x, anchor_y), 3, (255, 200, 0), -1)
            if alpha_mask is not None:
                cv2.circle(alpha_mask, (anchor_x, anchor_y), 3, 255, -1)

            # Draw label background
            cv2.rectangle(
                img_rgb,
                (text_x - 5, text_y - text_h - 5),
                (text_x + text_w + 5, text_y + 5),
                (50, 50, 50),
                -1,
            )
            if alpha_mask is not None:
                cv2.rectangle(
                    alpha_mask,
                    (text_x - 5, text_y - text_h - 5),
                    (text_x + text_w + 5, text_y + 5),
                    255,
                    -1,
                )

            # Draw label text
            cv2.putText(
                img_rgb,
                name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if alpha_mask is not None:
                cv2.putText(
                    alpha_mask,
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    255,
                    1,
                    cv2.LINE_AA,
                )

    # Recombine with alpha channel if needed
    if has_alpha:
        alpha_channel = np.maximum(alpha_channel, alpha_mask)
        img = cv2.merge(
            [img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2], alpha_channel]
        )

    cv2.imwrite(image_path, img)


def setup_camera(camera_state):
    """Setup camera based on camera state."""
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "MainCamera"
    bpy.context.scene.camera = cam

    mode = camera_state.get("mode", "ViewScene")
    args = camera_state.get("args", {})

    # Get relative transformations (these apply to any base camera mode)
    h_rot = camera_state.get("horizontal_rotation", 0.0)
    v_rot = camera_state.get("vertical_rotation", 0.0)
    movements = camera_state.get("movements", [])

    # Original ViewScene and FocusOn modes - determine base camera position
    view_type = args.get("view", "Iso")
    zoom = float(args.get("zoom", 1.0))
    target_name = args.get("target_name", None)

    # 1. Determine Target Center and Size
    min_b = Vector((float("inf"), float("inf"), float("inf")))
    max_b = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in bpy.context.scene.objects:
        ic(f"Scene Object: {obj.name}, Type: {obj.type}, Location: {obj.location}")
        ic(f"  Parent: {obj.parent.name if obj.parent else 'None'}")

    targets = []
    if mode == "FocusOn" and target_name:
        obj = bpy.data.objects.get(target_name)
        if obj:
            targets.append(obj)
            # Find the two nearest objects to provide context
            target_center = obj.location
            other_objects = []
            for other_obj in bpy.context.scene.objects:
                if (
                    other_obj.type == "EMPTY"
                    and "Floor" not in other_obj.name
                    and other_obj != obj
                    and other_obj not in targets
                ):
                    distance = (other_obj.location - target_center).length
                    other_objects.append((distance, other_obj))

            # Sort by distance and take the 2 closest
            other_objects.sort(key=lambda x: x[0])
            for dist, nearby_obj in other_objects[:2]:
                targets.append(nearby_obj)
                targets.extend(
                    [c for c in nearby_obj.children_recursive if c.type == "MESH"]
                )
    else:
        # ViewScene: All EMPTY root objects except infrastructure (consistent with loaded_names)
        for obj in bpy.context.scene.objects:
            if (
                obj.type == "EMPTY"
                and "Floor" not in obj.name
                and "Plane" not in obj.name
                and "CamTarget" not in obj.name
                and obj.parent is None  # Only root objects
            ):
                targets.append(obj)

    ic(targets)
    if not targets:
        center = Vector((0, 0, 0))
        size = 2.0
    else:
        has_bounds = False
        for obj in targets:
            # For EMPTY objects, get bounds from all child MESH objects
            mesh_objects = []
            if obj.type == "MESH":
                mesh_objects = [obj]
            else:
                # Get all MESH children recursively
                mesh_objects = [c for c in obj.children_recursive if c.type == "MESH"]

            for mesh_obj in mesh_objects:
                if hasattr(mesh_obj, "bound_box"):
                    has_bounds = True
                    for v in mesh_obj.bound_box:
                        world_v = mesh_obj.matrix_world @ Vector(v)
                        min_b.x = min(min_b.x, world_v.x)
                        min_b.y = min(min_b.y, world_v.y)
                        min_b.z = min(min_b.z, world_v.z)
                        max_b.x = max(max_b.x, world_v.x)
                        max_b.y = max(max_b.y, world_v.y)
                        max_b.z = max(max_b.z, world_v.z)

        if not has_bounds:
            center = Vector((0, 0, 0))
            size = 2.0
        else:
            center = (min_b + max_b) / 2.0
            size_vec = max_b - min_b
            size = max(size_vec.x, size_vec.y, size_vec.z)
            if size < 0.5:
                size = 0.5

    # 2. Setup Target Empty
    bpy.ops.object.empty_add(location=center)
    target_empty = bpy.context.active_object
    target_empty.name = "CamTarget"

    # 3. Calculate Base Offset (before relative transformations)
    base_dist = size * 2.5 * zoom
    if base_dist < 1.0:
        base_dist = 1.0

    # Determine base angles for each view type
    if view_type == "Top":
        base_h_angle = 0.0
        base_v_angle = 90.0
        distance = base_dist
    elif view_type == "Front":
        base_h_angle = 0.0
        base_v_angle = 0.0
        distance = base_dist
    elif view_type == "Side":
        base_h_angle = 90.0
        base_v_angle = 0.0
        distance = base_dist
    elif view_type == "Back":
        base_h_angle = 180.0
        base_v_angle = 0.0
        distance = base_dist
    else:  # "Iso" / Default
        base_h_angle = 45.0
        base_v_angle = 35.0
        distance = base_dist

    # 4. Apply relative rotations
    h_angle = math.radians(base_h_angle + h_rot)
    v_angle = math.radians(base_v_angle + v_rot)

    # Clamp vertical angle to avoid gimbal lock
    v_angle = max(math.radians(5), min(math.radians(85), v_angle))

    # 5. Calculate camera position using spherical coordinates
    x = distance * math.cos(v_angle) * math.sin(h_angle)
    y = -distance * math.cos(v_angle) * math.cos(h_angle)
    z = distance * math.sin(v_angle)

    cam.location = center + Vector((x, y, z))

    # 6. Apply accumulated movements
    for move in movements:
        direction = move["direction"]
        dist = move["distance"]

        # Calculate camera's local axes
        to_center = (center - cam.location).normalized()
        camera_right = to_center.cross(Vector((0, 0, 1))).normalized()
        camera_up = camera_right.cross(to_center).normalized()

        if direction == "Forward":
            cam.location += to_center * dist
        elif direction == "Backward":
            cam.location -= to_center * dist
        elif direction == "Left":
            cam.location -= camera_right * dist
        elif direction == "Right":
            cam.location += camera_right * dist
        elif direction == "Up":
            cam.location += camera_up * dist
        elif direction == "Down":
            cam.location -= camera_up * dist

    # 7. Track Constraints
    const = cam.constraints.new(type="TRACK_TO")
    const.target = target_empty
    const.track_axis = "TRACK_NEGATIVE_Z"
    const.up_axis = "UP_Y"

    return cam


def check_collisions(object_names, detailed=False):
    """Check for collisions between objects using BVH trees.

    Args:
        object_names: List of object names to check for collisions
        detailed: If False, returns simple string list (backward compatible).
                  If True, returns dict with collision normals and movement suggestions.

    Returns:
        If detailed=False: List of strings like ["TableA <-> ChairB", ...]
        If detailed=True: List of dicts with:
            - "pair": str, collision pair name
            - "collision_normal": Vector, averaged normal from object1's colliding polygons
            - "suggested_move_a": Vector, suggested movement for object1
            - "suggested_move_b": Vector, suggested movement for object2
    """
    import mathutils
    from mathutils import Vector

    collisions = []
    trees = {}
    mesh_data = {}  # Store polygon normals and centers for detailed mode
    move_distance = 0.1  # Configurable push-away distance

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # 1. Build BVH Trees for all objects in World Space
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if not obj:
            continue

        # If object is an Empty (common in GLTF root), try to find its first Mesh child
        target_obj = obj
        if obj.type != "MESH":
            mesh_children = [c for c in obj.children_recursive if c.type == "MESH"]
            if mesh_children:
                target_obj = mesh_children[0]
            else:
                continue

        obj_eval = target_obj.evaluated_get(depsgraph)
        try:
            mesh = obj_eval.to_mesh()
        except:
            continue

        if not mesh:
            continue

        mesh.transform(obj.matrix_world)

        try:
            bvh = mathutils.bvhtree.BVHTree.FromPolygons(
                [v.co for v in mesh.vertices], [p.vertices for p in mesh.polygons]
            )
            trees[name] = bvh

            # Store polygon normals and centers in world-space for detailed mode
            if detailed:
                poly_normals = []
                poly_centers = []
                for poly in mesh.polygons:
                    # Get polygon normal in world-space
                    poly_normals.append(poly.normal.copy())
                    # Get polygon center by averaging its vertices
                    vertices = [mesh.vertices[i].co for i in poly.vertices]
                    if vertices:
                        center = Vector((0.0, 0.0, 0.0))
                        for v in vertices:
                            center += v
                        center /= len(vertices)
                        poly_centers.append(center)
                mesh_data[name] = {
                    "poly_normals": poly_normals,
                    "poly_centers": poly_centers,
                }
        except Exception as e:
            print(f"Warning: Failed to create BVH for {name}: {e}")

        obj_eval.to_mesh_clear()

    # 2. Check Overlaps
    obj_list = list(trees.keys())
    for i in range(len(obj_list)):
        name1 = obj_list[i]
        for j in range(i + 1, len(obj_list)):
            name2 = obj_list[j]

            # Skip collision detection between walls
            if name1.startswith("Wall_") and name2.startswith("Wall_"):
                continue

            overlap = trees[name1].overlap(trees[name2])

            if overlap:
                if not detailed:
                    # Simple mode: backward compatible string format
                    collisions.append(f"{name1} <-> {name2}")
                else:
                    # Detailed mode: calculate collision normal and movement suggestions
                    # Average the normals of overlapping polygons from object1
                    avg_normal = Vector((0.0, 0.0, 0.0))
                    count = 0

                    if name1 in mesh_data:
                        poly_normals = mesh_data[name1]["poly_normals"]
                        # overlap returns list of (poly_idx_a, poly_idx_b) tuples
                        for poly_idx_a, _ in overlap:
                            if 0 <= poly_idx_a < len(poly_normals):
                                avg_normal += poly_normals[poly_idx_a]
                                count += 1

                    if count > 0:
                        avg_normal /= count
                        avg_normal.normalize()
                    else:
                        # Fallback: use a default up vector
                        avg_normal = Vector((0.0, 0.0, 1.0))

                    # Calculate movement suggestions
                    suggested_move_a = avg_normal * move_distance
                    suggested_move_b = -avg_normal * move_distance

                    collisions.append(
                        {
                            "pair": f"{name1} <-> {name2}",
                            "collision_normal": avg_normal,
                            "suggested_move_a": suggested_move_a,
                            "suggested_move_b": suggested_move_b,
                        }
                    )

    return collisions


def ensure_transparent_material(obj, alpha=0.3):
    """Ensure object has a transparent white material."""
    if obj.type != "MESH" or obj.data is None:
        return

    obj.data.materials.clear()

    mat = bpy.data.materials.new(name=f"TransparentWhite_{obj.name}")
    mat.diffuse_color = (0.95, 0.95, 0.95, alpha)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (200, 0)
    bsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    mat.blend_method = "BLEND"
    mat.shadow_method = "NONE"

    obj.data.materials.append(mat)


def add_light():
    """Add simple studio lighting setup for EEVEE renderer."""
    for obj in bpy.data.objects:
        if obj.name.startswith("Area"):
            return

    # Create key light (main light)
    bpy.ops.object.light_add(type="AREA", location=(4, 4, 5))
    key_light = bpy.context.view_layer.objects.active
    key_light.data.energy = 1000
    key_light.data.size = 5

    # Create fill light (soft light)
    bpy.ops.object.light_add(type="AREA", location=(-4, -4, 5))
    fill_light = bpy.context.view_layer.objects.active
    fill_light.data.energy = 500
    fill_light.data.size = 5

    # Create back light (rim light)
    bpy.ops.object.light_add(type="AREA", location=(0, 5, 5))
    back_light = bpy.context.view_layer.objects.active
    back_light.data.energy = 300
    back_light.data.size = 3


def render_scene(
    state,
    output_image_path,
    save_blend_path=None,
    render_mode="final",
):
    """Render the scene using BLENDER_EEVEE engine."""
    clean_scene()

    # Setup world with optional HDRI
    hdri_path = "/studio.exr"
    hdri_strength = 2.0
    setup_world(hdri_path=hdri_path, hdri_strength=hdri_strength)

    floor_texture = state.get("floor_texture")
    scene_objects = state.get("objects", {})
    camera_state = state.get("camera", {})

    loaded_names = []
    for name, data in scene_objects.items():
        mesh_path = data["mesh_path"]
        if not os.path.exists(mesh_path):
            continue

        bpy.ops.object.select_all(action="DESELECT")

        if mesh_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=mesh_path)
        elif mesh_path.endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=mesh_path)

        if not bpy.context.selected_objects:
            continue

        root_obj = None
        for obj in bpy.context.selected_objects:
            if obj.parent is None:
                root_obj = obj
                break

        if not root_obj:
            root_obj = bpy.context.selected_objects[0]

        root_obj.name = name
        root_obj.location = Vector(data["position"])
        rot = data["rotation"]

        root_obj.rotation_mode = "XYZ"
        root_obj.rotation_euler = Euler(
            (math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2])), "XYZ"
        )
        root_obj.scale = Vector(data["scale"])

        for obj in bpy.context.selected_objects:
            if name.startswith("Wall_"):
                ensure_transparent_material(obj, alpha=0.3)

        loaded_names.append(name)
    # Create floor texture if provided (cover object bounds only)
    if floor_texture and os.path.exists(floor_texture) and loaded_names:
        min_x, max_x, min_y, max_y = get_objects_bounds(loaded_names)
        create_floor_texture(floor_texture, min_x, max_x, min_y, max_y)

    cam = setup_camera(camera_state)

    # --- Fixed save logic ---
    if save_blend_path:
        # 1. Ensure an absolute output path.
        abs_path = os.path.abspath(save_blend_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # 2. Force dependency-graph update to materialize object data.
        bpy.context.view_layer.update()

        # 3. Save blend file; compress=True is recommended.
        # check_existing=False avoids overwrite prompts in headless runs.
        bpy.ops.wm.save_as_mainfile(
            filepath=abs_path, check_existing=False, compress=True
        )

        # 4. Critical: force filesystem flush to disk.
        # Standalone bpy may keep file locks briefly after script completion.
        try:
            with open(abs_path, "ab") as f:
                os.fsync(f.fileno())
            print(f"Blend file successfully flushed to: {abs_path}")
        except Exception as e:
            print(f"Sync error: {e}")

    bpy.context.view_layer.update()
    cam_matrix = cam.matrix_world.copy()

    # Calculate labels for HUD
    labels = []
    if render_mode == "agent":
        scene = bpy.context.scene
        for name in loaded_names:
            obj = bpy.data.objects.get(name)
            if obj:
                bbox_world = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
                max_z = max([v.z for v in bbox_world])
                center_x = sum([v.x for v in bbox_world]) / 8.0
                center_y = sum([v.y for v in bbox_world]) / 8.0

                label_pos = Vector((center_x, center_y, max_z + 0.2))
                co_2d = bpy_extras.object_utils.world_to_camera_view(
                    scene, cam, label_pos
                )

                if co_2d.z > 0:
                    labels.append({"name": name, "x": co_2d.x, "y": co_2d.y})

    collisions = check_collisions(loaded_names, detailed=False)

    # Use BLENDER_EEVEE for both agent and final modes
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 540
    bpy.context.scene.render.filepath = output_image_path

    # Transparent background
    bpy.context.scene.render.film_transparent = True

    # Add EEVEE lighting (simple fixed position lights complement HDRI)
    add_light()

    bpy.ops.render.render(write_still=True)

    # Only draw HUD for agent mode
    if render_mode == "agent":
        try:
            draw_hud_on_image(output_image_path, cam_matrix, labels)
        except Exception as e:
            print(f"Warning: Failed to draw HUD/Labels: {e}")

    return {"collisions": collisions}
