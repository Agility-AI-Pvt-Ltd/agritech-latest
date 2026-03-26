"""
pipeline/prompts/profile_prompt.py  –  Prompt for LLM-based user profile extraction.
"""

EXTRACT_SYSTEM = """You are a data extraction assistant.
Given a user message and the assistant reply, extract any personal or farm facts the user mentioned.
Return ONLY a valid JSON object (no markdown, no prose) with ONLY the keys that were explicitly mentioned.
Valid keys: name, language, location, latitude, longitude, farm_size_acres, soil_type, crops (list), extra_facts (dict).
CRITICAL: If the user provides ANY city, village, state, or address (even just replying to "Where are you?"), you MUST extract it as the `location` key!
Omit any key where no value was stated. Return {} if nothing new was mentioned."""
