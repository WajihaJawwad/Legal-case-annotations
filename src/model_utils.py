import json
from typing import Dict, Any

from transformers import pipeline  # Example for HF models

ANNOTATION_LABELS = [
    "Case Name",
    "Petitioner",
    "Respondent",
    "Issue Presented",
    "Facts of the Case",
    "Ruling by Lower Court",
    "Argument",
    "Statute",
    "Rules",
    "Precedent",
    "Ratio of the Decision",
    "Final Ruling"
]


def generate_annotation(case_text: str, model_pipeline, prompt_template: str) -> Dict[str, Any]:
    """
    Generate structured legal annotations from an LLM using JSON output.

    Args:
        case_text (str): The input legal case text.
        model_pipeline: Hugging Face pipeline (or other inference object).
        prompt_template (str): The JSON-enforcing prompt template.

    Returns:
        dict: Parsed annotation with all 12 fields.
    """
    # Build the prompt
    prompt = prompt_template.format(case_text=case_text)

    # Call the model
    response = model_pipeline(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
    raw_output = response[0]["generated_text"]

    # Try parsing JSON
    parsed = _safe_json_parse(raw_output)

    # Validate & fill missing fields
    validated = _validate_fields(parsed)

    return validated


def _safe_json_parse(output: str) -> Dict[str, Any]:
    """
    Safely parse JSON from model output.
    """
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        # Attempt to recover: strip extra text, find JSON substring
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            return json.loads(output[start:end])
        except Exception:
            # Return empty dict if unrecoverable
            return {}


def _validate_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all 12 annotation fields are present.
    Fill missing ones with empty strings.
    """
    validated = {}
    for label in ANNOTATION_LABELS:
        validated[label] = parsed.get(label, "").strip()
    return validated

