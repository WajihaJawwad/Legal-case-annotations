import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from data_utils import load_predex_dataset
from prompt_template import PROMPT_TEMPLATE
from evaluation import evaluate_annotations
from model_utils import chunk_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results/experiment2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = {
    "saullm-7b": {
        "name": "saul-ai/saullm-7b-chat",
        "max_tokens": 4096
    }
}


def annotate_with_model(model_key, cases):
    """Annotate cases with the given model."""
    model_name = MODEL[model_key]["name"]
    max_len = MODEL[model_key]["max_tokens"]

    print(f"\n--- Loading {model_key} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

    annotations = {}
    for case in tqdm(cases, desc=f"Annotating with {model_key}"):
        case_id = case["CaseName"]
        case_text = case["Input"]

        # Chunking with 20% overlap
        chunks = chunk_text(case_text, tokenizer, max_len, overlap=0.2)
        responses = []

        for chunk in chunks:
            prompt = PROMPT_TEMPLATE.format(case_text=chunk)
            output = generator(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )[0]["generated_text"]
            responses.append(output)

        # Merge responses if multiple chunks
        annotations[case_id] = " ".join(responses)

    # Save raw annotations
    with open(os.path.join(OUTPUT_DIR, f"{model_key}_annotations.json"), "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Save as CSV
    df = pd.DataFrame(list(annotations.items()), columns=["CaseName", f"{model_key}_annotation"])
    df.to_csv(os.path.join(OUTPUT_DIR, f"{model_key}_annotations.csv"), index=False)

    return annotations


def main():
    # Load dataset (150 cases from PreDeX)
    cases = load_predex_dataset()

    # Load reference annotations (gold standard)
    with open("data/reference_annotations.json", "r", encoding="utf-8") as f:
        reference_annotations = json.load(f)

    results = {}
    eval_results = {}

    for model_key in MODEL.keys():
        results[model_key] = annotate_with_model(model_key, cases)

        print(f"\nEvaluating {model_key} against reference annotations...")
        eval_results[model_key] = evaluate_annotations(results[model_key], reference_annotations)

    # Save evaluation results
    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print("\nExperiment 2 completed. Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
