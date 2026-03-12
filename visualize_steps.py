import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

from matplotlib.pyplot import draw
from PIL import Image, ImageDraw, ImageFont


def parse_reason_from_response(response_text):
    """
    Extracts reasoning from VLM response.
    Returns reason string, or None if not found.
    """
    reason_match = re.search(
        r"Reason:\s*(.*?)(?=Action:|$)", response_text, re.DOTALL | re.IGNORECASE
    )
    if reason_match:
        return reason_match.group(1).strip()
    return None


def parse_system_message_from_user(user_msg):
    """
    Extracts the System Messages section from a user message.
    Returns system message string, or None if not found or is "None".
    """
    if user_msg.get("role") != "user":
        return None

    content = user_msg.get("content", "")
    if not content:
        return None

    if isinstance(content, str):
        user_text = content
    elif isinstance(content, list):
        # Extract text from content list
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                user_text = item.get("text", "")
                break
        else:
            return None
    else:
        return None

    # Extract "# System Messages" section
    sys_pattern = r"# System Messages\s*\n(.+?)(?=\n#|$)"
    sys_match = re.search(sys_pattern, user_text, re.DOTALL)
    if sys_match:
        system_message = sys_match.group(1).strip()
        # Return None if it's "None" (no system message)
        if system_message == "None":
            return None
        return system_message
    return None


def parse_action_from_response(response_text):
    """
    Extracts action from (VM response.
    Returns action string (e.g., "Create('chair')"), or None if not found.
    """
    # Try JSON block first
    json_match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
    if json_match:
        try:
            json_content = json_match.group(1).strip()
            parsed = json.loads(json_content)
            # Handle multiple actions in the list
            if isinstance(parsed, list):
                actions = []
                for item in parsed:
                    if item and "type" in item and "args" in item:
                        action_type = item["type"]
                        args = item["args"]
                        args_str = ", ".join(
                            [
                                f"'{arg}'" if isinstance(arg, str) else str(arg)
                                for arg in args
                            ]
                        )
                        actions.append(f"{action_type}({args_str})")
                # Return all actions joined with newline for readability
                return "\n".join(actions) if actions else None
            # Handle single action (not in a list)
            elif parsed and "type" in parsed and "args" in parsed:
                action_type = parsed["type"]
                args = parsed["args"]
                args_str = ", ".join(
                    [f"'{arg}'" if isinstance(arg, str) else str(arg) for arg in args]
                )
                return f"{action_type}({args_str})"
        except:
            pass

    # Fallback: Action: Func(...) pattern
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\((.*)\)", response_text)
    if action_match:
        func_name = action_match.group(1)
        args_str = action_match.group(2).strip()
        if args_str.endswith(","):
            args_str = args_str[:-1]
        return f"{func_name}({args_str})"

    return None


def create_summary(output_dir):
    output_dir = Path(output_dir)

    # Find all steps, excluding verification images (step_*_verify.png)
    steps = sorted(
        [
            p
            for p in glob.glob(str(output_dir / "step_*.png"))
            if not p.endswith("_verify.png")
        ]
    )
    if not steps:
        print("No step images found.")
        return

    # Load chat history to extract reasons, actions, and system messages
    # step_info = {step_num: {"reason": str, "action": str, "system": str}}
    step_info = {}
    chat_history_path = output_dir / "logs" / "agent_loop" / "chat_history.json"
    if chat_history_path.exists():
        try:
            with open(chat_history_path, "r") as f:
                chat_history = json.load(f)
                # History format: [system, user, assistant, system, user, assistant, ...]
                # Use dynamic step_counter to handle verification messages (which don't have actions)
                step_counter = (
                    0  # Track actual step count (excluding verification messages)
                )
                i = 0
                while i < len(chat_history):
                    if i + 2 >= len(chat_history):
                        break

                    # Get system message from the user message before this assistant
                    system_message = parse_system_message_from_user(chat_history[i + 1])

                    # Get assistant message (contains actions)
                    assistant_msg = chat_history[i + 2]
                    if assistant_msg.get("role") == "assistant":
                        content = assistant_msg.get("content", "")
                        reason = parse_reason_from_response(content)
                        action = parse_action_from_response(content)

                        # Only increment step_counter if there are actual actions
                        # (skip verification messages that have no actions)
                        if action or reason:
                            step_counter += 1
                            step_info[step_counter] = {
                                "reason": reason or "",
                                "action": action or "",
                                "system": system_message,
                            }

                    i += 3
        except Exception as e:
            print(f"Warning: Could not load chat history: {e}")

    # Collect data
    step_data = []

    # Sort steps by number
    steps.sort(key=lambda p: int(Path(p).stem.split("_")[1]))

    for img_path in steps:
        step_num = int(Path(img_path).stem.split("_")[1])

        parts = []

        # Add system message (collision warning) if available
        if step_num in step_info and step_info[step_num]["system"]:
            parts.append(f"System: {step_info[step_num]['system']}")

        # Add reasoning if available
        if step_num in step_info and step_info[step_num]["reason"]:
            parts.append(f"Reason: {step_info[step_num]['reason']}")

        # Add action if available
        if step_num in step_info and step_info[step_num]["action"]:
            parts.append(f"Action: {step_info[step_num]['action']}")
        elif not parts:  # Only show "No Action" if there's no system message or reason
            parts.append("Init / No Action")

        step_data.append(
            {"img_path": img_path, "text": f"Step {step_num}:\n" + "\n".join(parts)}
        )

    # Create Grid Image
    num_images = len(step_data)
    cols = 4
    rows = math.ceil(num_images / cols)

    # Load first image to get dimensions
    sample_img = Image.open(step_data[0]["img_path"])
    w, h = sample_img.size

    # Target size for thumbnails (e.g. width 400)
    thumb_w = 800
    thumb_h = int(h * (thumb_w / w))

    padding = 30
    text_height = 400  # Increased height to display all text.

    grid_w = cols * thumb_w + (cols + 1) * padding
    grid_h = rows * (thumb_h + text_height) + (rows + 1) * padding

    # Create background
    summary_img = Image.new("RGB", (grid_w, grid_h), (220, 220, 220))
    draw = ImageDraw.Draw(summary_img)

    # Change this value to adjust font size.
    font = ImageFont.load_default(size=20)

    for idx, item in enumerate(step_data):
        row = idx // cols
        col = idx % cols

        x = padding + col * (thumb_w + padding)
        y = padding + row * (thumb_h + text_height + padding)

        img = Image.open(item["img_path"]).convert("RGBA")
        img.thumbnail((thumb_w, thumb_h))

        summary_img.paste(img, (x, y), mask=img.split()[3])

        # Draw text with a two-column layout to show more content.
        text = item["text"]

        # Wrap text intelligently into multiple lines.
        col_width = (thumb_w - 30) // 2  # Two columns with a center gap.
        lines = []

        for paragraph in text.split("\n"):
            if not paragraph.strip():
                continue

            # Simple width-based wrapping.
            words = paragraph.split(" ")
            current_line = ""

            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]

                if line_width <= col_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

        # Two-column layout without strict truncation.
        text_start_y = y + thumb_h + 10
        line_height = 22

        # Compute maximum lines per column.
        max_lines_per_col = (text_height - 20) // line_height

        # Left column.
        left_col_x = x + 5
        for i, line in enumerate(lines[:max_lines_per_col]):
            text_y = text_start_y + i * line_height
            draw.text((left_col_x, text_y), line, fill=(0, 0, 0), font=font)

        # Right column for remaining lines.
        if len(lines) > max_lines_per_col:
            right_col_x = x + col_width + 20
            for i, line in enumerate(lines[max_lines_per_col : max_lines_per_col * 2]):
                text_y = text_start_y + i * line_height
                draw.text((right_col_x, text_y), line, fill=(0, 0, 0), font=font)

    save_path = output_dir / "summary_grid.png"
    summary_img.save(save_path)
    print(f"Summary saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="outputs2/a_campsite_with_a_large_tent_a_camping_chair_a_cam_20260217_194726",
        type=str,
        help="Output directory to visualize",
    )
    args = parser.parse_args()
    create_summary(args.dir)
