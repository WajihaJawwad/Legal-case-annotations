from __future__ import annotations

from typing import Iterable, List
import pandas as pd


LABELS: List[str] = [
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
    "Ruling",
]


def load_predex_csv(path: str) -> pd.DataFrame:
    """Load PredEx CSV ensuring required columns exist."""
    df = pd.read_csv(path, encoding="latin-1")
    _ensure_columns(df, required=["Case Name", "Input"])
    return df


def load_predex_hf(split: str = "train") -> pd.DataFrame:
    """Load PredEx from Hugging Face into a pandas DataFrame."""
    from datasets import load_dataset  # local import to keep dependency optional

    ds = load_dataset("L-NLProc/PredEx", split=split)
    df = pd.DataFrame(ds)
    _ensure_columns(df, required=["Case Name", "Input"])
    return df


def sample_cases(df: pd.DataFrame, n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Sample N cases with non-empty Input."""
    work = df.dropna(subset=["Input"]).copy()
    if len(work) < n:
        raise ValueError(f"Requested {n} cases, but only {len(work)} available.")
    return work.sample(n=n, random_state=seed).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def chunk_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap: int = 128,
) -> List[str]:
    """Split text into overlapping token windows and decode back to strings."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap >= max_tokens:
        raise ValueError("overlap must be < max_tokens")

    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [""]

    stride = max_tokens - overlap
    chunks: List[str] = []
    for start in range(0, len(ids), stride):
        end = min(start + max_tokens, len(ids))
        chunk_ids = ids[start:end]
        if not chunk_ids:
            break
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(ids):
            break
    return chunks


def chunk_dataframe(
    df: pd.DataFrame,
    tokenizer,
    max_tokens: int,
    overlap: int = 128,
    input_col: str = "Input",
    case_col: str = "Case Name",
) -> pd.DataFrame:
    """Return a long-form DataFrame with one row per chunk."""
    _ensure_columns(df, required=[case_col, input_col])

    rows = []
    for i, row in df.iterrows():
        case = str(row[case_col])
        text = str(row[input_col])
        parts = chunk_by_tokens(text, tokenizer, max_tokens=max_tokens, overlap=overlap)
        for j, p in enumerate(parts):
            rows.append(
                {
                    case_col: case,
                    "chunk_id": j,
                    "n_chunks": len(parts),
                    "chunk_text": p,
                }
            )
    return pd.DataFrame(rows)


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
