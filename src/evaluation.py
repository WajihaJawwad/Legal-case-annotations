import numpy as np
from typing import Dict, List

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from blanc import BlancHelp

# Initialize global scorers
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
BLANC_HELP = BlancHelp()


def compute_metrics_per_label(ref: str, pred: str) -> Dict[str, float]:

    metrics = {}

    # Rouge scores
    rouge_scores = ROUGE_SCORER.score(ref, pred)
    metrics["ROUGE-1"] = rouge_scores["rouge1"].fmeasure
    metrics["ROUGE-2"] = rouge_scores["rouge2"].fmeasure
    metrics["ROUGE-L"] = rouge_scores["rougeL"].fmeasure

    # BLEU
    smoothie = SmoothingFunction().method4
    metrics["BLEU"] = sentence_bleu(
        [ref.split()], pred.split(), smoothing_function=smoothie
    )

    # METEOR
    try:
        metrics["METEOR"] = meteor_score([ref], pred)
    except Exception:
        metrics["METEOR"] = 0.0

    # BERTScore (average F1 across tokens)
    try:
        P, R, F1 = bert_score([pred], [ref], lang="en", verbose=False)
        metrics["BERTScore"] = F1.mean().item()
    except Exception:
        metrics["BERTScore"] = 0.0

    # BLANC
    try:
        metrics["BLANC"] = BLANC_HELP.eval_once(ref, pred)["f1"]
    except Exception:
        metrics["BLANC"] = 0.0

    return metrics


def compute_metrics_per_case(ref_case: Dict[str, str], pred_case: Dict[str, str]) -> Dict[str, float]:
    """
    Compute average metrics across all 12 labels for one case.
    """
    all_label_metrics = []

    for label, ref_text in ref_case.items():
        pred_text = pred_case.get(label, "")
        label_metrics = compute_metrics_per_label(ref_text, pred_text)
        all_label_metrics.append(label_metrics)

    # Average across labels
    averaged = {metric: np.mean([m[metric] for m in all_label_metrics]) for metric in all_label_metrics[0]}
    return averaged


def compute_overall_metrics(ref_cases: List[Dict[str, str]], pred_cases: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Compute overall metrics averaged across all cases.
    """
    assert len(ref_cases) == len(pred_cases), "Mismatch in number of cases."

    case_metrics = []
    for ref, pred in zip(ref_cases, pred_cases):
        case_result = compute_metrics_per_case(ref, pred)
        case_metrics.append(case_result)

    # Final average across cases
    overall = {metric: np.mean([cm[metric] for cm in case_metrics]) for metric in case_metrics[0]}
    return overall
