import json
import os

from openai import OpenAI


class LLM:
    def __init__(self, system_prompt, save_dir, model="o1-mini-2024-09-12", history=None):
        """
        Initializes the LLM with a system prompt and optional conversation history.

        Parameters:
        - system_prompt: The system message to provide context for the LLM.
        - save_dir: Directory where conversation history will be saved.
        - history: Optional conversation history to continue from.
        """
        self.client = OpenAI()
        self.model = model
        # Cannot give both system prompt and history
        assert not (system_prompt and history), "Cannot give both system prompt and history"
        self.history = history or [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def prompt(self, user_input):
        """
        Sends a user prompt to the LLM, receives the response, and updates the conversation history.

        Parameters:
        - user_input: The text prompt from the user.

        Returns:
        - The LLM's response text.
        """
        # Append user message to history
        self.history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            # response_format={"type": "text"},
            # temperature=1,
            # max_completion_tokens=2048,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
        )

        # Extract assistant response
        assistant_response = response.choices[0].message.content

        # Append assistant response to history
        self.history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})

        return assistant_response

    def save_history(self, filename="conversation_history.json"):
        """
        Saves conversation history to a JSON file.

        Parameters:
        - filename: The name of the file to save the history to.

        Returns:
        - str: Path where the history was saved, or None if saving failed
        """
        save_path = os.path.join(self.save_dir, filename)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)
            return save_path
        except Exception as e:
            print(f"Failed to save conversation history: {e}")
            return None
