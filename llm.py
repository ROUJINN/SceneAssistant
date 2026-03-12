import base64
import datetime
import json
import os
import time
from abc import ABC, abstractmethod

from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Logger / Recorder
# Every chat call is logged by recorder, including prompt and model response.
# self.history_file points to chat_history.json.
class ChatRecorder:
    def __init__(self, run_id=None, base_dir="logs"):
        if run_id is None:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_id = run_id
        self.log_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history_file = os.path.join(self.log_dir, "chat_history.json")
        print(f"Chat session logs saved to: {self.log_dir}")

    def save_history(self, messages):
        """Save the full structured history to JSON."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save history: {e}")

    def load_history(self):
        """Load structured history from JSON."""
        if os.path.exists(path=self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load history: {e}")
        return []

    def trim_history_to_step(self, step_num):
        """Trim chat_history.json for resuming at step_num.

        When resuming at step N, we want to keep steps 1 to N-1,
        because step N will be re-executed.

        Each step consists of: 1 system message + 1 user message + 1 assistant message.
        Verification steps (with "You are a scene verification agent." system prompt)
        must be skipped as they are not counted in the step numbering.
        """
        if not os.path.exists(self.history_file):
            print(f"No history file found at {self.history_file}")
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                messages = json.load(f)

            messages_to_keep = []
            step_count = 0
            target_steps = step_num - 1  # Keep steps 1 to N-1

            i = 0
            while i < len(messages) and step_count < target_steps:
                # Need at least 3 messages for a complete step
                if i + 2 >= len(messages):
                    break

                system_msg = messages[i]
                user_msg = messages[i + 1]
                assistant_msg = messages[i + 2]

                # Check if this is a verification step
                is_verify = False
                if system_msg.get("role") == "system":
                    for item in system_msg.get("content", []):
                        if (
                            item.get("type") == "text"
                            and "You are a scene verification agent."
                            in item.get("text", "")
                        ):
                            is_verify = True
                            break

                if is_verify:
                    # Skip verification step
                    i += 3
                else:
                    # Include regular step
                    messages_to_keep.extend([system_msg, user_msg, assistant_msg])
                    step_count += 1
                    i += 3

            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(messages_to_keep, f, indent=2, default=str)

            print(
                f"Trimmed chat_history.json to keep steps 1-{step_count} ({len(messages_to_keep)} messages)"
            )
        except Exception as e:
            print(f"Failed to trim history: {e}")


# Abstract Base Class for LLMs
class BaseLLM(ABC):
    def __init__(self, recorder: ChatRecorder):
        self.recorder = recorder

    @abstractmethod
    def chat(self, user_text=None, image_path=None, system_text=None):
        pass

    @abstractmethod
    def reload_history(self):
        """Reload history from recorder into memory"""
        pass


# OpenAI-compatible Implementation (e.g., Qwen, DeepSeek)
class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key,
        base_url,
        model_name,
        system_prompt="You are a helpful assistant.",
        recorder=None,
        history=None,
    ):
        super().__init__(recorder)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        # Initialize with history if provided, otherwise start with system prompt
        if history:
            self.messages = history
        else:
            self.messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]

    def chat(self, user_text=None, image_path=None, system_text=None):
        # --- Construct Prompt for THIS turn only (Stateless) ---
        api_messages = []

        # 1. System Prompt
        # Use provided system_text, or fall back to the initial one if needed
        current_system_text = (
            system_text if system_text else "You are a helpful assistant."
        )
        api_messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": current_system_text}],
            }
        )

        # 2. User Content
        user_content = []

        # Normalize image_path to list
        if image_path and isinstance(image_path, str):
            image_path = [image_path]

        if image_path:
            for img in image_path:
                image_base64 = encode_image(img)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    }
                )

        if user_text is None:
            user_text = ""
        user_content.append({"type": "text", "text": user_text})

        api_messages.append({"role": "user", "content": user_content})

        # --- Call API with retry mechanism ---
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name, messages=api_messages
                )
                ai_response = completion.choices[0].message.content
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached.")
                    raise  # Re-raise the exception

        # --- Update Logging History (Stateful) ---
        # Record the complete api_messages (system + user) + assistant response
        self.messages.extend(api_messages)
        self.messages.append({"role": "assistant", "content": ai_response})

        # Save state
        self.recorder.save_history(self.messages)

        return ai_response

    def reload_history(self):
        """Reload history from recorder into memory"""
        loaded_history = self.recorder.load_history()
        if loaded_history:
            self.messages = loaded_history
            print(f"Reloaded {len(self.messages)} messages into OpenAI LLM")


# Google Gemini Implementation
class GeminiLLM(BaseLLM):
    def __init__(self, api_key, model_name, recorder=None, history=None):
        super().__init__(recorder)
        self.model_name = model_name
        self.client = genai.Client(
            api_key=api_key,
            http_options={"base_url": "https://api.zhizengzeng.com/google"},
        )

        self.json_history = []

        if history:
            self.json_history = history
            # History reconstruction logic skipped as we are now stateless
            pass

        # Stateless mode: No chat session needed
        # self.chat_session = ...

    def chat(self, user_text=None, image_path=None, system_text=None):
        content_items = []
        user_json_content = []

        # 1. Process images first (normalize image_path to list).
        if image_path:
            if isinstance(image_path, str):
                image_path = [image_path]
            for path in image_path:
                image = Image.open(path)
                content_items.append(image)
                user_json_content.append({"type": "image_ref", "path": path})

        # 2. Process text afterwards.
        final_text = user_text if user_text else ""
        if final_text:
            content_items.append(final_text)
            user_json_content.append({"type": "text", "text": final_text})

        # Configure Generation
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            system_instruction=system_text,
        )

        # Stateless call with retry mechanism
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=content_items, config=config
                )
                ai_response = response.text
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached.")
                    raise

        # Update local JSON history (Logging only)
        # Record system message + user + assistant
        if system_text:
            self.json_history.append(
                {"role": "system", "content": [{"type": "text", "text": system_text}]}
            )
        self.json_history.append({"role": "user", "content": user_json_content})
        self.json_history.append({"role": "assistant", "content": ai_response})

        # Save state
        self.recorder.save_history(self.json_history)

        return ai_response

    def reload_history(self):
        """Reload history from recorder into memory"""
        loaded_history = self.recorder.load_history()
        if loaded_history:
            self.json_history = loaded_history
            print(f"Reloaded {len(self.json_history)} messages into Gemini LLM")


# Factory / Wrapper Class
class LLM:
    def __init__(self, model="qwen", run_id=None, resume=False, log_dir="logs"):
        self.model = model

        # Initialize Recorder
        self.recorder = ChatRecorder(run_id=run_id, base_dir=log_dir)

        # Try load history if resume is requested
        history = []
        if resume:
            history = self.recorder.load_history()
            if history:
                print(
                    f"Resumed session {self.recorder.run_id} with {len(history)} messages."
                )
        if self.model == "gemini":
            self.delegate = GeminiLLM(
                api_key=os.getenv("GOOGLE_API_KEY"),
                model_name="gemini-3-flash-preview",
                recorder=self.recorder,
                history=history,
            )
        elif self.model == "qwen":
            model_name = "kimi-k2.5"
            self.delegate = OpenAILLM(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_name=model_name,
                recorder=self.recorder,
                history=history,
            )
            print(f"using {model_name}")
        else:
            raise ValueError(f"Model {self.model} not supported")

    def chat(self, user_text=None, image_path=None, system_text=None):
        return self.delegate.chat(
            user_text=user_text, image_path=image_path, system_text=system_text
        )

    def reload_history(self):
        """Reload history from recorder into memory"""
        self.delegate.reload_history()


if __name__ == "__main__":
    # Simple test
    llm = LLM(model="gemini", resume=False)
    response = llm.chat(user_text="Hello, can you help me?")
    print("AI Response:", response)
    # response = llm.chat(user_text="Hello, can you help me?")
    # print("AI Response:", response)
