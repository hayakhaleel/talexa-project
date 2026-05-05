PROMPT = """
You are given a presentation slide image.

Task:
Find the EXACT location of the text: "{focus}"

Rules:
- Return ONLY one point (x, y)
- The point must be at the START of the text (the LEFTMOST character of the phrase)
- If the text is multi-line, choose the start of the FIRST line
- If the text is centered, choose the left edge of the text
- Coordinates MUST be normalized between 0 and 1
- (0,0) is top-left, (1,1) is bottom-right
- Do NOT return the center
- Do NOT return bounding boxes

Respond ONLY like:
(0.12, 0.34)
"""
