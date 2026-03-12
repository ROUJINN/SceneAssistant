"""Convert chat_history.json to chat_readable.txt format."""

import argparse
import json
import os
import sys


def format_content_for_readable(content):
    """Format message content for readable output."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        image_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_ref":
                    image_parts.append(item.get("path", ""))
                elif item.get("type") == "image_url":
                    # Extract path from data URL if possible
                    url = item.get("image_url", {}).get("url", "")
                    if "base64" in url:
                        image_parts.append("[base64 image]")
                    else:
                        image_parts.append(url)
            elif isinstance(item, str):
                text_parts.append(item)

        result = ""
        if text_parts:
            result += " ".join(text_parts)
        if image_parts:
            result += "\nImages: " + ", ".join(image_parts)
        return result

    return str(content)


def convert_chat_to_readable(dir_path):
    """Convert chat_history.json to chat_readable.txt."""
    log_dir = os.path.join(dir_path, "logs", "agent_loop")
    chat_history_path = os.path.join(log_dir, "chat_history.json")
    readable_path = os.path.join(log_dir, "chat_readable.txt")

    if not os.path.exists(chat_history_path):
        print(f"Error: chat_history.json not found at {chat_history_path}")
        return False

    try:
        with open(chat_history_path, "r", encoding="utf-8") as f:
            messages = json.load(f)

        with open(readable_path, "w", encoding="utf-8") as f:
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "system":
                    f.write(f"System: {format_content_for_readable(content)}\n")
                    f.write("-" * 40 + "\n")
                elif role == "user":
                    formatted = format_content_for_readable(content)
                    f.write(f"User: {formatted}\n")
                elif role == "assistant":
                    f.write(f"\nAI response: {content}\n")
                    f.write("-" * 40 + "\n")

        print(f"Generated chat_readable.txt at {readable_path}")
        return True

    except Exception as e:
        print(f"Error converting chat history: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert chat_history.json to chat_readable.txt"
    )
    parser.add_argument("--dir", required=True, help="output dir")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: Directory not found: {args.dir}")
        sys.exit(1)

    success = convert_chat_to_readable(args.dir)
    sys.exit(0 if success else 1)
