from __future__ import annotations

from typing import Dict, Iterable, List
from src.dataset_utils import LABELS

PROMPT_TEMPLATE = """
You are a legal expert tasked with generating structured and detailed annotations for a given legal case. 
Follow the structured format below to ensure clarity and consistency.

Case Details:
{case_text}

Annotation Structure:

Case Name:Official name of the case.
Petitioner: The party initiating the case.
Respondent: The opposing party.
Issue Presented: Key legal question(s) addressed.
Facts of the Case: Relevant background and context.
Ruling by Lower Court: Decision made by lower courts before reaching this stage.
Argument: Main legal arguments from both petitioner and respondent.
Statute: Legal statutes referenced in the case.
Rules:** Applicable legal or procedural rules.
Precedent: Past cases influencing the decision.
Ratio of the Decision: Core legal reasoning.
Final Ruling: Final decision of the court.

Response Guidelines:
Maintain a neutral, professional tone.
Use formal legal terminology.
Base annotations strictly on case facts and legal principles.
"""



def build_annotation_prompt(case_text: str) -> str:
    """Return the full structured prompt for a case."""
    return PROMPT_TEMPLATE.format(case_text=case_text)


def merge_label_dicts(partials: List[Dict[str, str]], labels: Iterable[str] = LABELS) -> Dict[str, str]:
    merged: Dict[str, List[str]] = {k: [] for k in labels}
    for d in partials:
        for k in labels:
            v = (d.get(k) or "").strip()
            if v:
                merged[k].append(v)
    return {k: " ".join(merged[k]).strip() for k in labels}
