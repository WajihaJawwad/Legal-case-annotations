import os
import json
import random
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from peft import LoraConfig, get_peft_model

from data_utils import load_predex_dataset
from prompt_template import PROMPT_TEMPLATE
from evaluation import evaluate_annotations
from model_utils import chunk_text

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results/experiment3"
CHECKPOINT_DIR = "checkpoints/llama2-7b_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MAX_LEN = 4096


def finetune_llama50():
    with open("data/annotations_50.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    cases_list = list(cases.values())
    random.shuffle(cases_list)
    split_idx = int(0.8 * len(cases_list))
    train_cases, test_cases = cases_list[:split_idx], cases_list[split_idx:]

    train_dataset = Dataset.from_list([{"text": c} for c in train_cases])
    test_dataset = Dataset.from_list([{"text": c} for c in test_cases])
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"])

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )

    print("\n--- Initial Evaluation ---")
    print(trainer.evaluate())

    print("\n--- Training ---")
    trainer.train()

    print("\n--- Final Evaluation ---")
    print(trainer.evaluate())

    trainer.save_model(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

def annotate_with_finetuned(cases, model_path, label="llama2-7b-finetuned"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

    annotations = {}
    for case in tqdm(cases, desc=f"Annotating with {label}"):
        case_id = case["CaseName"]
        case_text = case["Input"]

        chunks = chunk_text(case_text, tokenizer, MAX_LEN, overlap=0.2)
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

        annotations[case_id] = " ".join(responses)

    with open(os.path.join(OUTPUT_DIR, f"{label}_annotations.json"), "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(list(annotations.items()), columns=["CaseName", f"{label}_annotation"])
    df.to_csv(os.path.join(OUTPUT_DIR, f"{label}_annotations.csv"), index=False)

    return annotations


def main():
    finetune_llama50()

    cases = load_predex_dataset()
    finetuned_outputs = annotate_with_finetuned(cases, CHECKPOINT_DIR, "llama2-7b-finetuned")

    with open("data/reference_annotations.json", "r", encoding="utf-8") as f:
        reference_annotations = json.load(f)

    print("\nEvaluating finetuned LLaMA2 against reference annotations...")
    eval_results = evaluate_annotations(finetuned_outputs, reference_annotations)

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print("\nExperiment 3 completed. Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

