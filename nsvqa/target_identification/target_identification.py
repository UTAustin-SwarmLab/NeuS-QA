import datetime
import json
from pathlib import Path

from tqdm import tqdm

from nsvqa.target_identification.llm import LLM


def clean_and_parse_json(raw_str):
    start = raw_str.find("{")
    end = raw_str.rfind("}") + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)


def process_datapoint(llm, question, candidates, specification):
    # Construct the prompt for the LLM
    prompt = f"""Now that we have identified the specification and its temporal window in the video, let's determine where to look for the answer to the question.

The specification we identified earlier occurs between start_time and end_time seconds in the video. Based on the temporal relationship in the question, we need to determine the target frame window where the answer is most likely to be found.

Please determine the target frame window by considering:
1. If the question asks about events "after" the specification, look slightly before the end of the specification and extend into the future
2. If the question asks about events "before" the specification, look slightly after the start of the specification and extend into the past
3. If the question asks about events "during" the specification, focus on the middle portion of the specification window

IMPORTANT: The target window MUST ALWAYS include the entire identified window (start_time to end_time) plus additional time before or after:
- For "after" questions: Include the entire identified window plus 5-10 seconds after
- For "before" questions: Include the entire identified window plus 5-10 seconds before
- For "during" questions: Include the entire identified window plus 2-3 seconds on either side

The target window should be wide enough to capture the visual changes described in the candidates, but not so wide as to include irrelevant content.

Provide your response in the following format:
Target frame window: "[target_start_time, target_end_time]" (make sure to include the quotes and square brackets)
Explanation: "[brief explanation of why you chose these timestamps, referencing both the temporal relationship in the question and the type of visual changes we're looking for]"

Example:
For a question asking "what happens after X" where X occurs between start_time and end_time seconds and the candidates involve camera view changes:
Target frame window: "[start_time, end_time + 10]"
Explanation: Since the question asks about events after the specification and the candidates involve camera view changes, we include the entire identified window (start_time to end_time) and extend 10 seconds after to ensure we capture the complete camera view change.

Inputs:
Question: {question}
Specification: {specification}
Identified temporal window for the specification: [start_time, end_time]
Possible answers (candidates):
{chr(10).join(f"- {candidate}" for candidate in candidates)}

Expected Output (only output the following JSON structure — nothing else):
{{
  "target_frame_window": "[...]",
  "explanation": "..."
}}
"""

    # Get LLM response
    response = llm.prompt(prompt)
    response = clean_and_parse_json(response)
    target_frame_window = response["target_frame_window"]
    explanation = response["explanation"]

    # Save the conversation history with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = llm.save_history(f"conversation_history_{timestamp}.json")

    return {
        "frame_window": target_frame_window,
        "explanation": explanation,
        "saved_path": history_path,
    }


def identify_target(question, candidates, specification, conversation_history):
    # Read the conversation history
    history_path = conversation_history
    with open(history_path, "r") as f:
        history = json.load(f)
    llm = LLM(
        system_prompt=None,
        save_dir="outputs/3_target_identification/longvideobench",
        model="o1-mini-2024-09-12",
        history=history,
    )

    # Get target identification results
    result = process_datapoint(llm, question, candidates, specification)
    return result

