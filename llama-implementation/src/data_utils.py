"""
Data loading and formatting for MTS-Dialog.

MTS-Dialog  → local dataset/MTS-Dialog/
  task: dialogue → (section_header, section_text)
  splits: train 1201 / validation 100 / test1 200 / test2 200
  augmented training: 3 back-translation files (~8400 additional pairs)

"""

import json
import os
from functools import partial
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

# ── Prompt / response templates ───────────────────────────────────────────────

SYSTEM_MTS = (
    "You are a clinical documentation assistant specializing in medical transcription. "
    "Given a doctor-patient dialogue, identify the correct clinical note section and write an "
    "accurate section summary in formal clinical documentation style.\n\n"
    "General guidelines:\n"
    "- Use third-person perspective and formal clinical sentences\n"
    "- Include ALL clinically relevant details from the dialogue for this section\n"
    "- Convert spoken numbers and dates to numerals (e.g. 'fifty five' → '55', 'January eighth' → '01/08')\n"
    "- Use standard clinical abbreviations where appropriate (q.d., b.i.d., p.r.n., h/o, s/p, HTN, DM, etc.)\n\n"
    "Section-specific format rules — follow these precisely:\n\n"
    "CC (Chief Complaint): 2–8 word noun phrase only. State the presenting complaint as briefly as possible. "
    "No demographics, no narrative, no background history. "
    "Examples: 'Chest pain.', 'Fever of unknown origin.', 'Left wrist pain.'\n\n"
    "GENHX (General History / History of Present Illness): Full narrative paragraph, typically 80–130 words. "
    "Must include: patient age and gender, reason for visit, relevant medical background, and a detailed "
    "account of the presenting illness including onset, duration, and associated symptoms. "
    "Begin with 'The patient is a [age]-year-old [gender] who...' or similar. "
    "Do NOT write a one-line summary — this section always requires a full paragraph.\n\n"
    "ASSESSMENT: Numbered list of diagnoses or clinical impressions. "
    "Format as: '1. Condition. 2. Condition.' Do NOT write a narrative paragraph. "
    "For a single diagnosis, a short descriptive phrase is acceptable.\n\n"
    "PASTMEDICALHX (Past Medical History): Concise list of known medical conditions. "
    "Can be sentence form or abbreviated list (e.g. 'Significant for HTN, DM, and COPD.'). "
    "If unchanged from prior visit, write 'Otherwise reviewed and noted.' or 'Reviewed and unchanged.'\n\n"
    "FAM/SOCHX (Family and Social History): Document both family medical history AND social history "
    "(occupation, living situation, marital status, tobacco/alcohol/drug use). "
    "Can be combined in one or two sentences. "
    "Examples: 'HTN, father with SLE.', 'Married, works as a nurse. Denies tobacco or ETOH use.'\n\n"
    "ROS (Review of Systems): System-by-system review using 'Positive for...' / 'Negative for...' "
    "or system headers (e.g. 'CARDIOVASCULAR: No chest pain or palpitations. MSK: Negative myalgia.'). "
    "List all systems addressed in the dialogue.\n\n"
    "MEDICATIONS: List all medications mentioned. Include drug name, dose, route, and frequency where "
    "stated in the dialogue. Include relevant context (e.g. 'recently started', 'discontinued'). "
    "Do not list only names — preserve the full clinical detail.\n\n"
    "ALLERGY: State drug/food allergies or negation. "
    "Examples: 'No known drug allergies.', 'Allergic to penicillin.', 'None.'\n\n"
    "PASTSURGICAL (Past Surgical History): List prior surgeries, ideally with approximate date or age. "
    "Example: 'Appendectomy at age 21. C-section 8 years ago.' If none: 'Negative.' or 'None.'\n\n"
    "EXAM (Physical Examination): Document exam findings by body system using system headers "
    "(e.g. 'CHEST: Lungs clear to auscultation. ABDOMEN: Soft, non-tender.'). "
    "Report what was found, not what was asked.\n\n"
    "DIAGNOSIS: Single-line clinical diagnosis. A medical term or brief phrase. "
    "Examples: 'Migraine with aura.', 'Diarrhea.', 'Aftercare following motor vehicle accident.'\n\n"
    "EDCOURSE (Emergency Department / Hospital Course): Narrative of what occurred during the ED visit "
    "or hospital stay — treatments administered, procedures performed, patient response, and current status. "
    "Write complete sentences. Typically 40–120 words. "
    "Example: 'The patient was given IV morphine 4 mg with significant improvement in discomfort.'\n\n"
    "DISPOSITION: Very brief — patient's discharge status or plan. "
    "Examples: 'Stable.', 'The patient will be going home.', 'To home with his son.', 'Fair, but improved.'\n\n"
    "PLAN: Treatment and follow-up plan. Can be a numbered list or short sentences. "
    "Include medications to continue, referrals, activity instructions, and follow-up timing.\n\n"
    "IMAGING: Report the imaging modality and specific findings. "
    "Examples: 'Chest x-ray revealed diffuse pulmonary edema.', 'X-ray shows no bony abnormality.'\n\n"
    "LABS: Report lab test results with specific values where mentioned. "
    "Example: 'Sodium 133, Creatinine 0.2. Cardiac enzymes negative. BUN within normal limits.'\n\n"
    "IMMUNIZATIONS: One short phrase. Examples: 'Up-to-date.', 'None.', 'Not sure.'\n\n"
    "GYNHX (Gynecological History): OB/GYN history including gravida/para status, menstrual history, "
    "and relevant gynecological procedures. Example: 'G2P2. Last menstrual period 3 weeks ago.'\n\n"
    "PROCEDURES: Brief list of procedures performed or planned. "
    "Examples: 'Permanent pacemaker placement.', 'None.'\n\n"
    "OTHER_HISTORY: Any other history not captured elsewhere — can include changes since last visit, "
    "social/family context, or a combined catch-all review. Typically very brief (1–2 sentences).\n\n"
    "Respond with exactly:\n"
    "Section: <section header>\n"
    "Summary: <section text>"
)

SYSTEM_ACI = (
    "You are a clinical documentation assistant. "
    "Given a doctor-patient conversation transcript, write a complete clinical visit note. "
    "Include all relevant sections such as CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
    "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, and ASSESSMENT AND PLAN. "
    "Be concise, accurate, and clinically appropriate."
)


def _mts_user(dialogue: str) -> str:
    return (
        f"Dialogue:\n{dialogue}\n\n"
        "Identify the clinical note section this dialogue belongs to, then write a detailed "
        "section summary that captures all relevant clinical information from the dialogue."
    )


def _mts_assistant(header: str, text: str) -> str:
    return f"Section: {header}\nSummary: {text}"


def _aci_user(dialogue: str) -> str:
    return f"Transcript:\n{dialogue}\n\nWrite the complete clinical visit note."


# ── MTS-Dialog ────────────────────────────────────────────────────────────────

_MTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset", "MTS-Dialog")

_MTS_SPLIT_FILES = {
    "train":      "Main-Dataset/MTS-Dialog-TrainingSet.csv",
    "validation": "Main-Dataset/MTS-Dialog-ValidationSet.csv",
    "test":       "Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
    "test1":      "Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
    "test2":      "Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv",
}

_MTS_AUG_FILES = [
    "Augmented-Data/MTS-Dialog-Augmented-TrainingSet-1-En-FR-EN-2402-Pairs.csv",
    "Augmented-Data/MTS-Dialog-Augmented-TrainingSet-2-EN-ES-EN-2402-Pairs.csv",
    "Augmented-Data/MTS-Dialog-Augmented-TrainingSet-3-FR-and-ES-3603-Pairs-final.csv",
]


def load_mts_dialog(split: str) -> list[dict]:
    """Load a local MTS-Dialog CSV split and return a list of row dicts."""
    if split not in _MTS_SPLIT_FILES:
        raise ValueError(f"Unknown MTS split '{split}'. Choose from: {list(_MTS_SPLIT_FILES)}")
    path = os.path.join(_MTS_ROOT, _MTS_SPLIT_FILES[split])
    df = pd.read_csv(path)
    return df.to_dict("records")


def load_mts_augmented() -> list[dict]:
    """Load all three augmented back-translation training CSVs and return combined row dicts."""
    rows = []
    for rel_path in _MTS_AUG_FILES:
        path = os.path.join(_MTS_ROOT, rel_path)
        df = pd.read_csv(path)
        rows.extend(df.to_dict("records"))
    return rows


def mts_to_examples(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append({
            "task": "mts",
            "dialogue": row["dialogue"],
            "section_header": row["section_header"],
            "section_text": row["section_text"],
            "user": _mts_user(row["dialogue"]),
            "assistant": _mts_assistant(row["section_header"], row["section_text"]),
        })
    return out


# ── ACI-Bench ─────────────────────────────────────────────────────────────────

_ACI_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "aci", "data", "challenge_data_json"
)

_ACI_SPLIT_MAP = {
    "train": "train",
    "valid": "valid",
    "test1": "clinicalnlp_taskB_test1",
    "test2": "clinicalnlp_taskC_test2",
    "test3": "clef_taskC_test3",
}

ACI_SECTIONS = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]


def load_aci_bench(split: str = "train", note_type: str = "full") -> list[dict]:
    """
    split:     'train' | 'valid' | 'test1' | 'test2' | 'test3'
    note_type: 'full'  | 'subjective' | 'objective_exam' | 'objective_results' | 'assessment_and_plan'
    """
    if split not in _ACI_SPLIT_MAP:
        raise ValueError(f"Unknown ACI split '{split}'. Choose from: {list(_ACI_SPLIT_MAP)}")
    suffix = "" if note_type == "full" else f"_{note_type}"
    path = os.path.join(_ACI_ROOT, f"{_ACI_SPLIT_MAP[split]}{suffix}.json")
    with open(path) as f:
        return json.load(f)["data"]


def aci_to_examples(items: list[dict], note_type: str = "full") -> list[dict]:
    out = []
    for item in items:
        out.append({
            "task": "aci",
            "note_type": note_type,
            "dialogue": item["src"],
            "note": item["tgt"],
            "file": item.get("file", ""),
            "user": _aci_user(item["src"]),
            "assistant": item["tgt"],
        })
    return out


# ── Chat-template helpers ─────────────────────────────────────────────────────

def build_chat_text(
    tokenizer: PreTrainedTokenizer,
    system: str,
    user: str,
    assistant: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt
    )


# ── Tokenized PyTorch dataset ─────────────────────────────────────────────────

class SummarizationDataset(TorchDataset):
    """
    Tokenizes instruction examples for causal LM training.
    Labels are -100 on the prompt (system + user) tokens so only the completion is supervised.
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        system = SYSTEM_MTS if ex["task"] == "mts" else SYSTEM_ACI

        full_text = build_chat_text(
            self.tokenizer, system, ex["user"], assistant=ex["assistant"]
        )
        prompt_text = build_chat_text(
            self.tokenizer, system, ex["user"], add_generation_prompt=True
        )

        full_ids = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True, return_tensors="pt"
        )["input_ids"].squeeze(0)

        prompt_len = self.tokenizer(
            prompt_text, max_length=self.max_length, truncation=True, return_tensors="pt"
        )["input_ids"].shape[1]

        labels = full_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        n = b["input_ids"].shape[0]
        pad = max_len - n
        input_ids.append(torch.cat([b["input_ids"], b["input_ids"].new_full((pad,), pad_token_id)]))
        attention_mask.append(torch.cat([b["attention_mask"], b["attention_mask"].new_zeros(pad)]))
        labels.append(torch.cat([b["labels"], b["labels"].new_full((pad,), -100)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def make_collate_fn(pad_token_id: int):
    return partial(collate_fn, pad_token_id=pad_token_id)


# ── High-level loaders ────────────────────────────────────────────────────────

def get_train_examples(
    use_mts: bool = True,
    use_aci: bool = True,
    use_mts_augmented: bool = True,
) -> list[dict]:
    examples = []
    if use_mts:
        examples.extend(mts_to_examples(load_mts_dialog("train")))
        if use_mts_augmented:
            examples.extend(mts_to_examples(load_mts_augmented()))
    if use_aci:
        examples.extend(aci_to_examples(load_aci_bench("train", "full")))
    return examples


def get_val_examples(dataset: str = "mts") -> list[dict]:
    if dataset == "mts":
        return mts_to_examples(load_mts_dialog("validation"))
    return aci_to_examples(load_aci_bench("valid", "full"))


def get_test_examples(
    dataset: str = "mts",
    mts_split: str = "test1",
    aci_split: str = "test1",
) -> list[dict]:
    if dataset == "mts":
        return mts_to_examples(load_mts_dialog(mts_split))
    return aci_to_examples(load_aci_bench(aci_split, "full"))
