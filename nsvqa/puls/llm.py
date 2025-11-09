import datetime
import json
import os


class LLM:
    def __init__(self, client, save_dir="outputs/1_puls/longvideobench/conversation_history"):
        """Initialize LLM"""
        self.client = client
        self.history = []
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def prompt(self, p):
        """Send a prompt to the LM and update conversation history"""
        user_message = {"role": "user", "content": [{"type": "text", "text": p}]}
        self.history.append(user_message)

        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=self.history,
            store=False,
        )
        assistant_response = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        self.history.append(assistant_message)

        return assistant_response

    def save_history(self, filename="conversation_history.json"):
        """Save conversation history to a JSON file and return the save path"""
        if not self.save_dir:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, extension = os.path.splitext(filename)
        timestamped_filename = f"{base_name}_{timestamp}{extension}"

        save_path = os.path.join(self.save_dir, timestamped_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        return save_path

