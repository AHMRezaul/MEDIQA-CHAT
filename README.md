# MEDIQA-Chat 2023 Task A — Clinical Dialogue Summarization

End-to-end pipeline for section-aware clinical note generation from doctor-patient dialogues, covering QLoRA fine-tuning, retrieval-augmented generation (RAG), and automated evaluation.

---

## Table of Contents

1. [Task Definition](#1-task-definition)
2. [Dataset](#2-dataset)
3. [Directory Structure](#3-directory-structure)
4. [Preprocessing](#4-preprocessing)
5. [Prompt Design](#5-prompt-design)
6. [Model Architecture and Fine-Tuning (QLoRA)](#6-model-architecture-and-fine-tuning-qlora)
7. [Retrieval-Augmented Generation (RAG)](#7-retrieval-augmented-generation-rag)
8. [Inference](#8-inference)
9. [Post-processing](#9-post-processing)
10. [Evaluation](#10-evaluation)
11. [Experiments and Results](#11-experiments-and-results)
12. [SLURM Pipeline](#12-slurm-pipeline)
13. [Quickstart](#13-quickstart)

---

## 1. Task Definition

**MEDIQA-Chat 2023 Task A: Section-Aware Clinical Note Generation**

Given a short doctor-patient dialogue snippet, the model must produce two outputs:

- **Section header** — which part of a clinical note the dialogue belongs to (e.g., `chief complaint`, `history of present illness`, `past medical history`, `assessment and plan`, etc.)
- **Section text** — a formal, prose-style clinical summary of the dialogue content appropriate for that section

The test set contains **200 examples**. Evaluation uses section header exact-match accuracy plus NLG metrics (ROUGE, BERTScore, BLEURT) on the section text.

---

## 2. Dataset

**MTS-Dialog** (primary training source):

| Split | Examples |
|---|---|
| Train | 1,201 |
| Validation | 100 |
| Test 1 (MEDIQA-Chat 2023) | 200 |
| Test 2 (MEDIQA-Sum 2023) | 200 |

**Augmented data** (available but disabled in final configuration):
Three back-translation files (~8,400 additional English↔French↔Spanish↔English pairs) were evaluated but ultimately disabled because they raised validation loss — the translated text introduced noise that degraded in-domain English performance.

Dataset files are located in `dataset/MTS-Dialog/`.

---

## 3. Directory Structure

```
mediqa-chat/
├── configs/
│   └── train_config.yaml       # Training hyperparameters
├── dataset/
│   ├── MTS-Dialog/
│   │   ├── Main-Dataset/       # train / validation / test CSVs
│   │   └── Augmented-Data/     # back-translation augmentation CSVs
│   └── aci/                    # ACI-Bench data (Task B)
├── results/                    # Prediction CSVs and score JSONs
├── retrieval_index/            # Serialized RAG index (.pkl files)
├── runs/                       # LoRA checkpoints (best/ + epoch_N/)
├── scripts/
│   └── run_pipeline.slurm      # SLURM job script
├── src/
│   ├── data_utils.py           # Data loading, prompt templates, tokenized dataset
│   ├── train.py                # QLoRA fine-tuning
│   ├── inference.py            # Batched generation (zero-shot / finetuned / RAG)
│   ├── retrieval.py            # Dense retriever (build index + retrieve)
│   └── scorer.py               # ROUGE / BERTScore / BLEURT evaluation
└── chat-env/                   # Python virtual environment
```

---

## 4. Preprocessing

No explicit text cleaning was applied to the raw dialogues. Preprocessing was confined to formatting:

- Each training example was converted into a structured `{system, user, assistant}` three-turn chat format using the model's own **chat template** (`tokenizer.apply_chat_template`)
- For training, both the full sequence (prompt + completion) and the prompt-only sequence were tokenized; **labels were set to -100 on all prompt tokens** so the cross-entropy loss only supervises the model on generating the assistant response, not on re-predicting the input
- Padding direction: **right** during training (conventional for causal LM training), **left** during inference (required for batched generation so all sequences attend to real tokens at the right end)
- Sequences truncated to **2,048 tokens** during training; inference prompts truncated to **3,072 tokens** (to accommodate RAG context)

---

## 5. Prompt Design

The same prompt structure was used for both models across all experiment types. The system prompt contains general guidelines followed by per-section format rules covering all 20 possible section headers (~680 tokens total).

### System Prompt

```
You are a clinical documentation assistant specializing in medical transcription.
Given a doctor-patient dialogue, identify the correct clinical note section and write an
accurate section summary in formal clinical documentation style.

General guidelines:
- Use third-person perspective and formal clinical sentences
- Include ALL clinically relevant details from the dialogue for this section
- Convert spoken numbers and dates to numerals (e.g. 'fifty five' → '55', 'January eighth' → '01/08')
- Use standard clinical abbreviations where appropriate (q.d., b.i.d., p.r.n., h/o, s/p, HTN, DM, etc.)

Section-specific format rules — follow these precisely:

CC (Chief Complaint): 2–8 word noun phrase only. State the presenting complaint as briefly as possible.
No demographics, no narrative, no background history.
Examples: 'Chest pain.', 'Fever of unknown origin.', 'Left wrist pain.'

GENHX (General History / History of Present Illness): Full narrative paragraph, typically 80–130 words.
Must include: patient age and gender, reason for visit, relevant medical background, and a detailed
account of the presenting illness including onset, duration, and associated symptoms.
Begin with 'The patient is a [age]-year-old [gender] who...' or similar.
Do NOT write a one-line summary — this section always requires a full paragraph.

ASSESSMENT: Numbered list of diagnoses or clinical impressions.
Format as: '1. Condition. 2. Condition.' Do NOT write a narrative paragraph.
For a single diagnosis, a short descriptive phrase is acceptable.

PASTMEDICALHX (Past Medical History): Concise list of known medical conditions.
Can be sentence form or abbreviated list (e.g. 'Significant for HTN, DM, and COPD.').
If unchanged from prior visit, write 'Otherwise reviewed and noted.' or 'Reviewed and unchanged.'

FAM/SOCHX (Family and Social History): Document both family medical history AND social history
(occupation, living situation, marital status, tobacco/alcohol/drug use).
Can be combined in one or two sentences.
Examples: 'HTN, father with SLE.', 'Married, works as a nurse. Denies tobacco or ETOH use.'

ROS (Review of Systems): System-by-system review using 'Positive for...' / 'Negative for...'
or system headers (e.g. 'CARDIOVASCULAR: No chest pain or palpitations. MSK: Negative myalgia.').
List all systems addressed in the dialogue.

MEDICATIONS: List all medications mentioned. Include drug name, dose, route, and frequency where
stated in the dialogue. Include relevant context (e.g. 'recently started', 'discontinued').
Do not list only names — preserve the full clinical detail.

ALLERGY: State drug/food allergies or negation.
Examples: 'No known drug allergies.', 'Allergic to penicillin.', 'None.'

PASTSURGICAL (Past Surgical History): List prior surgeries, ideally with approximate date or age.
Example: 'Appendectomy at age 21. C-section 8 years ago.' If none: 'Negative.' or 'None.'

EXAM (Physical Examination): Document exam findings by body system using system headers
(e.g. 'CHEST: Lungs clear to auscultation. ABDOMEN: Soft, non-tender.').
Report what was found, not what was asked.

DIAGNOSIS: Single-line clinical diagnosis. A medical term or brief phrase.
Examples: 'Migraine with aura.', 'Diarrhea.', 'Aftercare following motor vehicle accident.'

EDCOURSE (Emergency Department / Hospital Course): Narrative of what occurred during the ED visit
or hospital stay — treatments administered, procedures performed, patient response, and current status.
Write complete sentences. Typically 40–120 words.
Example: 'The patient was given IV morphine 4 mg with significant improvement in discomfort.'

DISPOSITION: Very brief — patient's discharge status or plan.
Examples: 'Stable.', 'The patient will be going home.', 'To home with his son.', 'Fair, but improved.'

PLAN: Treatment and follow-up plan. Can be a numbered list or short sentences.
Include medications to continue, referrals, activity instructions, and follow-up timing.

IMAGING: Report the imaging modality and specific findings.
Examples: 'Chest x-ray revealed diffuse pulmonary edema.', 'X-ray shows no bony abnormality.'

LABS: Report lab test results with specific values where mentioned.
Example: 'Sodium 133, Creatinine 0.2. Cardiac enzymes negative. BUN within normal limits.'

IMMUNIZATIONS: One short phrase. Examples: 'Up-to-date.', 'None.', 'Not sure.'

GYNHX (Gynecological History): OB/GYN history including gravida/para status, menstrual history,
and relevant gynecological procedures. Example: 'G2P2. Last menstrual period 3 weeks ago.'

PROCEDURES: Brief list of procedures performed or planned.
Examples: 'Permanent pacemaker placement.', 'None.'

OTHER_HISTORY: Any other history not captured elsewhere — can include changes since last visit,
social/family context, or a combined catch-all review. Typically very brief (1–2 sentences).

Respond with exactly:
Section: <section header>
Summary: <section text>
```

### User Turn (no RAG)

```
Dialogue:
{doctor-patient dialogue text}

Identify the clinical note section this dialogue belongs to, then write a detailed
section summary that captures all relevant clinical information from the dialogue.
```

### Expected Assistant Response

```
Section: {section_header}
Summary: {section_text}
```

---

### Full Prompt Seen by the LLM (No RAG)

Using the model's chat template, the complete token sequence during training is (simplified):

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a clinical documentation assistant specializing in medical transcription ...
[general guidelines + 20 section-specific format rules]
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Dialogue:
Doctor: How are you feeling today?
Patient: I've been having chest pain for 3 days...

Identify the clinical note section this dialogue belongs to, then write a detailed
section summary that captures all relevant clinical information from the dialogue.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Section: GENHX
Summary: The patient is a 45-year-old male presenting with a 3-day history of chest pain ...
<|eot_id|>
```

*(Exact special tokens differ by model family; Qwen uses `<|im_start|>` / `<|im_end|>` markers instead.)*

During **inference**, the assistant turn is omitted and `add_generation_prompt=True` appends the assistant header token so the model generates from that position. The cross-entropy loss during training is masked (set to −100) on all system and user tokens — the model is only supervised on the assistant response.

---

### Full Prompt Seen by the LLM (With RAG, k=5)

Retrieved examples are prepended in the user turn as few-shot demonstrations, each formatted as a `Dialogue / Section / Summary` triple and separated by `---`:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a clinical documentation assistant specializing in medical transcription ...
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Dialogue:
[Retrieved example 1 — most similar training dialogue]
Section: GENHX
Summary: The patient presents with a 2-week history of...

---

Dialogue:
[Retrieved example 2]
Section: GENHX
Summary: Patient reports progressive shortness of breath...

---

[... 3 more retrieved examples (top-5 by cosine similarity) ...]

---

Dialogue:
[Test dialogue — the one to summarize]

Identify the clinical note section this dialogue belongs to, then write a detailed
section summary that captures all relevant clinical information from the dialogue.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

---

## 6. Model Architecture and Fine-Tuning (QLoRA)

Two large instruction-tuned models (~70B parameters each) were fine-tuned using the same methodology:

- **Model 1**: Llama-3.3-70B-Instruct
- **Model 2**: Qwen2.5-72B-Instruct

### Quantization

Both models were loaded in **4-bit NF4 quantization** (QLoRA) using BitsAndBytes:

| Setting | Value |
|---|---|
| Quantization type | NF4 (Normal Float 4) |
| Compute dtype | bfloat16 |
| Double quantization | Enabled |

**Why QLoRA**: A 70B model in full bf16 requires ~140 GB VRAM — beyond the capacity of available A100 80GB GPUs. 4-bit quantization reduces this to ~35 GB for the frozen weights, leaving headroom for activations and the trainable LoRA parameters.

### LoRA Configuration

LoRA (Low-Rank Adaptation) was applied to the attention and feed-forward projection matrices:

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Bias | none |

**Why these modules**: All seven projection matrices cover both the self-attention mechanism (q/k/v/o) and the SwiGLU feed-forward block (gate/up/down). Adapting all of them gives the model more capacity to restructure its representations for clinical summarization style, compared to adapting attention layers alone.

With r=16, each adapted matrix adds two low-rank matrices (A: d×r, B: r×d). For a 70B model this totals roughly 250–350M trainable parameters out of ~70B frozen — under 0.5% of total parameters.

### Training Hyperparameters

| Hyperparameter | Llama-3.3-70B | Qwen2.5-72B |
|---|---|---|
| Epochs | 3 | 3 |
| Learning rate | 1e-4 | 1e-4 |
| Per-device batch size | 1 | 1 |
| Gradient accumulation | 2 | 2 |
| GPUs | 4 × A100 80GB | 2 × A100 80GB |
| Effective batch size | 8 | 4 |
| Weight decay | 0.01 | 0.01 |
| Warmup ratio | 5% of total steps | 5% of total steps |
| LR schedule | Cosine decay | Cosine decay |
| Gradient norm clip | 1.0 | 1.0 |
| Max sequence length | 2,048 tokens | 2,048 tokens |

**Distributed training**: HuggingFace `accelerate` with mixed-precision `bf16` was used across multiple GPUs. `prepare_model_for_kbit_training` enabled gradient checkpointing and fp32 layer norms, required for stable QLoRA training.

**Checkpoint selection**: Validation loss was computed on the 100-example validation split after each epoch. The checkpoint with the **lowest validation loss** was saved as `runs/mts_lora/best/` and used for all inference and evaluation.

---

## 7. Retrieval-Augmented Generation (RAG)

### Knowledge Base

The retrieval index was built from the **1,201 MTS-Dialog training examples**. Each indexed entry stores the full dialogue text and its gold (section_header, section_text) pair.

### Embedding Model

Dialogues were encoded with **`sentence-transformers/all-MiniLM-L6-v2`**, a lightweight 22M-parameter bi-encoder producing 384-dimensional embeddings. All embeddings were **L2-normalized** so that the dot product equals cosine similarity.

**Why this encoder**: Fast (embeds 1,201 examples in seconds on CPU), produces semantically meaningful sentence embeddings, and is well-suited to matching similar conversational contexts without requiring medical domain-specific pretraining.

### Retrieval Criterion

At inference time, the test dialogue is encoded with the same encoder. **Cosine similarity** is computed between the test embedding and all 1,201 training embeddings (a single matrix multiplication). The **top-k most similar training dialogues** are returned (k=5 in primary RAG experiments).

The retrieval criterion is: *find training dialogues that are topically and linguistically most similar to the test dialogue.* The intuition is that a dialogue about similar symptoms or procedures will share a section header and similar clinical phrasing, providing the model with concrete stylistic demonstrations.

### RAG Context Injection

Retrieved examples are prepended **in the user turn** (not in the system prompt) as few-shot demonstrations. Each retrieved example is formatted as:

```
Dialogue:
{retrieved_dialogue}
Section: {retrieved_section_header}
Summary: {retrieved_section_text}
```

Multiple examples are separated by `---` and followed by the actual test input.

### Index Persistence

The retriever (embeddings + example metadata) is serialized to `retrieval_index/mts_train.pkl` after being built, so it does not need to be recomputed for each experiment.

---

## 8. Inference

**Batched greedy decoding** was used for all experiments:

| Setting | Value |
|---|---|
| Batch size | 8 |
| Decoding | Greedy (do_sample=False) |
| Max new tokens | 512 |
| Input truncation | 3,072 tokens |
| Padding | Left-padding |

For the **finetuned mode**, LoRA adapter weights were **merged into the base model** before inference (`merge_and_unload()`). This avoids per-forward-pass adapter overhead and makes generation speed identical to the base model.

Max new tokens was set to 512 after discovering ~10% of predictions were being cut off with the initial value of 256.

---

## 9. Post-processing

The model is prompted to respond in a fixed two-line format. After generation, a simple **line-by-line parser** extracts the outputs:

1. Scan each line for a prefix `"section:"` (case-insensitive) → extract the header
2. Scan each line for a prefix `"summary:"` (case-insensitive) → extract the text
3. Fallback: if neither prefix is found, the first line becomes the header, everything after becomes the text

No further normalization is applied before writing predictions to CSV. The scorer normalizes to lowercase only for header accuracy computation.

---

## 10. Evaluation

Four metrics were computed on the 200-example test set:

| Metric | What it measures |
|---|---|
| **Section Header Accuracy** | Exact match (case-insensitive) between predicted and reference section header |
| **ROUGE-1/2/L** | N-gram overlap between predicted and reference section text |
| **BERTScore F1** | Soft token-level semantic similarity using `microsoft/deberta-xlarge-mnli` contextual embeddings |
| **BLEURT-20** | Learned regression metric trained to predict human quality judgments |

**Aggregate score** = mean(ROUGE-1, BERTScore F1, BLEURT-20) — the primary ranking metric used in MEDIQA-Chat 2023.

Metrics are computed on section text only. Header accuracy is reported separately.

### Known Compatibility Issues Fixed

**BERTScore + transformers 5.x**: `bert_score` v0.3.x internally calls `build_inputs_with_special_tokens` on the tokenizer to detect special token positions. This method was removed from all tokenizer classes in transformers 5.x. Fix: monkey-patch a lambda directly onto the tokenizer instance at scorer initialization.

**DeBERTa tokenizer overflow**: `deberta-xlarge-mnli`'s tokenizer reports `model_max_length=1e30`, which overflows a Rust integer when passed to the tokenizer backend. Fix: clamp `tok.model_max_length = 512`.

Both fixes are implemented in `src/scorer.py`.

---

## 11. Experiments and Results

All scores are computed on the 200-example MTS-Dialog test set (Test 1). Best value per column among all systems is **bold**. Aggregate = mean(ROUGE-1, BERTScore F1, BLEURT) — the primary MEDIQA-Chat 2023 ranking metric.

| Model | Setup | Aggregate | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Header Acc. |
|---|---|---|---|---|---|---|---|
| Llama-3.3-70B | Zero-shot | 0.4908 | 0.3142 | 0.1458 | 0.2561 | 0.6393 | 67.0% |
| Llama-3.3-70B | Zero-shot + RAG | 0.4903 | 0.3126 | 0.1463 | 0.2554 | 0.6393 | 67.0% |
| Llama-3.3-70B | QLoRA FT | **0.5901** | **0.4618** | **0.2032** | **0.3911** | **0.7386** | **78.0%** |
| Llama-3.3-70B | QLoRA FT + RAG | 0.5051 | 0.3508 | 0.1654 | 0.3026 | 0.6530 | 62.0% |
| Qwen2.5-72B | Zero-shot | 0.5318 | 0.3757 | 0.1823 | 0.3025 | 0.6856 | 71.5% |
| Qwen2.5-72B | Zero-shot + RAG | 0.4991 | 0.3441 | 0.1571 | 0.2891 | 0.6308 | 62.0% |
| Qwen2.5-72B | QLoRA FT | 0.5582 | 0.4231 | 0.1933 | 0.3624 | 0.7190 | 77.0% |
| Qwen2.5-72B | QLoRA FT + RAG | 0.5133 | 0.3642 | 0.1664 | 0.3192 | 0.6764 | 66.0% |
| Flan-T5-Large | Published SOTA | 0.5790 | 0.4150 | — | — | 0.7070 | 78.0% |

*(ZS = zero-shot, FT = QLoRA fine-tuned, RAG = retrieval-augmented with top-5 MiniLM-retrieved training examples)*

### Key Observations

- **Llama-3.3-70B QLoRA FT surpasses the published SOTA**: aggregate 0.5901 vs Flan-T5-Large 0.5790 (+1.1pp), with higher ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, and matching header accuracy (78.0%), despite Flan-T5-Large being a task-specific supervised model
- **Fine-tuning is the single biggest lever**: both models gain ~+0.09–0.13 aggregate over their zero-shot baseline, and header accuracy rises from ~67–71% to 77–78%
- **RAG consistently hurts fine-tuned models**: Llama FT drops from 0.5901 to 0.5051 with RAG (−8.5pp); Qwen FT drops from 0.5582 to 0.5133 (−4.5pp). The retrieved few-shot examples introduce mixed section headers that conflict with the model's already well-calibrated section predictions. Header accuracy also falls (Llama: 78% → 62%, Qwen: 77% → 66%)
- **RAG also fails for zero-shot Llama**: aggregate drops marginally (0.4908 → 0.4903) with essentially no effect on header accuracy (67.0% both), suggesting zero-shot Llama is insensitive to retrieved demonstrations
- **Qwen zero-shot is notably strong** (0.5318 aggregate, 71.5% header acc), outperforming Llama zero-shot (0.4908, 67.0%) and approaching Flan-T5 SOTA without any fine-tuning, indicating stronger base clinical language priors
- **Qwen FT (0.5582) falls short of SOTA** despite fine-tuning, suggesting Llama-3.3-70B adapts more efficiently to the QLoRA regime for this task
- **BERTScore is the most stable metric**: the gap between best (Llama FT 0.7386) and worst (Llama ZS 0.6393) is ~0.10, versus ~0.13 for ROUGE-1, indicating models differ more in surface n-gram overlap than in semantic similarity to the reference

---

## 12. SLURM Pipeline

The full pipeline is orchestrated as a parameterized SLURM job (`scripts/run_pipeline.slurm`) with six stages:

| Stage | What runs |
|---|---|
| `build_index` | Encode training dialogues → save `.pkl` retrieval index |
| `train` | QLoRA fine-tuning with accelerate |
| `infer` | Inference only (requires existing checkpoint) |
| `eval` | Evaluation only (requires existing predictions CSV) |
| `infer_eval` | Inference + evaluation (most common post-training stage) |
| `rag_eval` | Build index (if needed) + RAG inference + evaluation |
| `all` | Build index → train → RAG inference → evaluation |

All stages are controlled by environment variables passable via `sbatch --export`:

| Variable | Default | Description |
|---|---|---|
| `STAGE` | `infer_eval` | Pipeline stage to run |
| `TASK` | `mts` | Task: `mts` or `aci` |
| `MODE` | `finetuned` | Inference mode: `zero_shot`, `finetuned`, `retrieval` |
| `MODEL_ID` | `meta-llama/Llama-3.3-70B-Instruct` | Base model HuggingFace ID |
| `MODEL_DIR` | `runs/mts_lora/best` | Path to LoRA checkpoint |
| `SPLIT` | `test` | Dataset split |
| `RETRIEVAL_INDEX` | *(empty)* | Path to `.pkl` index; enables RAG when set |
| `TOP_K` | `5` | Number of retrieved examples |
| `PREDICTIONS` | auto-named | Output CSV path |
| `SCORES` | auto-named | Output JSON path |

Example overrides:

```bash
# Run only inference + evaluation on test set with an existing checkpoint
sbatch --export=ALL,STAGE=infer_eval,MODEL_DIR=runs/mts_lora/best scripts/run_pipeline.slurm

# Run zero-shot inference with RAG (no fine-tuned checkpoint needed)
sbatch --export=ALL,STAGE=rag_eval,MODE=zero_shot,MODEL_DIR="" scripts/run_pipeline.slurm

# Evaluate an already-generated predictions file
sbatch --export=ALL,STAGE=eval,PREDICTIONS=results/my_preds.csv scripts/run_pipeline.slurm
```

---

## 13. Quickstart

### Environment Setup

```bash
source /scratch/akarim9/mediqa-chat/chat-env/bin/activate
export HF_HOME=/scratch/akarim9/.cache
export HF_TOKEN=<your_token>
export PYTHONPATH=/scratch/akarim9/mediqa-chat/src:$PYTHONPATH
```

### Build Retrieval Index (run once)

```bash
python src/retrieval.py build --task mts --index_path retrieval_index/mts_train.pkl
```

### Fine-Tune

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 src/train.py --config configs/train_config.yaml --load_in_4bit
```

### Run Inference

```bash
# Finetuned, no RAG
python src/inference.py --task mts --mode finetuned --model_dir runs/mts_lora/best \
  --split test --output_file results/mts_finetuned_test_preds.csv --max_new_tokens 512

# Zero-shot with RAG
python src/inference.py --task mts --mode zero_shot \
  --retrieval_index retrieval_index/mts_train.pkl --top_k 5 \
  --split test --output_file results/mts_zs_rag_test_preds.csv --max_new_tokens 512
```

### Evaluate

```bash
python src/scorer.py --task mts \
  --predictions results/mts_finetuned_test_preds.csv \
  --output results/mts_finetuned_test_scores.json
```

### Run Full Pipeline via SLURM

```bash
sbatch scripts/run_pipeline.slurm
```
