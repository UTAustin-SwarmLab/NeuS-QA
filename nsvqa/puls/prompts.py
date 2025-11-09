def find_prompt(prompt):
    full_prompt = f"""
You are an intelligent agent designed to extract structured representations from video question prompts. You will operate in two stages: (1) proposition extraction and (2) TL specification generation.

Stage 1: Proposition Extraction

Given an input question about a video, extract the atomic propositions that describe the underlying events or facts explicity referenced in the question. These propositions should describe object-action or object-object relationships stated in the question — avoid making assumptions or inferring any additional events. Avoid TL keywords such as 'and', 'or', 'not', 'until'.
Do not include ambiguous propositions that lack specificity. For instance, phrases like "guy does something" are ambiguous and should be omitted. Instead, focus on concrete actions or relationships. For example, given the prompt "In a bustling park, a child kicks a ball. What happens when the ball hits the bench?", the correct propositions are ["child kicks ball", "ball hits bench"].
If a proposition mentions subtitles/captions, the format of the proposition is the word "subtile_" followed by the subtitle in single quotes. Do NOT add words like "appears"/"says"/"mentions" after the subtitle; follow this format to create the individual proposition. For example, given the prompt "After the man gets up, what happens after the subtitle 'Hello Mr. Anderson' appeared?", the correct propositions are ["man gets up", "subtitle_'Hello Mr. Anderson'"].

Stage 2: TL Specification Generation

Using only the list of the propositions extracted in Stage 1, generate a single Temporal Logic (TL) specification that catpures the sequence of logical structure implied by the question. 

Rules:
- The formula must use each proposition **exactly once**
- Use only the TL operators: `AND`, `OR`, `NOT`, `UNTIL`
- Do **not** infer new events or rephrase propositions.
- The formula should reflect the temporal or logical relationships between the propositions under which the question would be understandable.

**Examples**

Example 1: "In a sunny meadow, a child plays with a kite and runs around. What does the child do after falling?"
Output:
{{
  "proposition": ["child plays with kite", "child runs around", "child falls"],
  "specification": "(child plays with kite AND child runs around) UNTIL child falls"
}}

Example 2: "In a dimly lit room, two robots stand silently. What happens when the red robot starts blinking or the green robot does not turn off?"
Output:
{{
  "proposition": ["robots stand silently", "red robot starts blinking", "green robot turns off"],
  "specification": "robots stand silently UNTIL (red robot starts blinking OR NOT green robot turns off)"
}}

Example 3: "Inside a cave, a man holds a lantern. What happens when the man sees the dragon?"
Output:
{{
  "proposition": ["man holds lantern", "man sees dragon"],
  "specification": "man holds lantern UNTIL man sees dragon"
}}

Example 4: "What happened on the screen before a man in black armor with glasses spoke into the microphone in front of a golden sky and the subtitles said 'country uh so we've seen significant'?"
Output:
{{
  "proposition": ["man in black armor with glasses spoke into the microphone", "man is in front of a golden sky", "subtitle_'country uh so we've seen significant'"],
  "specification": "(man in black armor with glasses spoke into the microphone AND man is in front of a golden sky) UNTIL (subtitle_'country uh so we've seen significant')"
}}

Example 5: "A news anchor with curled hair is wearing a pink blazer over a black base and sitting in front of the camera reading the news. What happened before the caption 'standards our climate editor Justin rout' appeared?"
Output:
{{
  "proposition": ["news anchor with curled hair is wearing a pink blazer over a black base", "news anchor sitting in front of the camera reading the news", "subtitle_'standards our climate editor Justin rout'"],
  "specification": "(news anchor with curled hair is wearing a pink blazer over a black base AND news anchor sitting in front of the camera reading the news) UNTIL subtitle_'standards our climate editor Justin rout'"
}}


Example 6: "How did the girl feel before turning on the computer?"
Output:
{{
  "proposition": ["girl turns on computer"],
  "specification": "(girl turns on computer)"
}}

**Now process the following prompt:**
Input:
{{
  "prompt": "{prompt}"
}}

Expected Output (only output the following JSON structure — nothing else):
{{
  "proposition": [...],
  "specification": "..."
}}
"""
    return full_prompt

