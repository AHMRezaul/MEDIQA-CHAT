# Error Analysis: Llama-3.3-70B FT vs Qwen2.5-72B FT

Systematic analysis of fine-tuned model predictions on the MTS-Dialog Task A test set (200 examples, `test1` split).
Both models evaluated in `infer_eval` mode (no RAG), using their respective best checkpoints.

- **Llama**: `runs/mts_lora/best` — job 7216485 — aggregate **0.5903**, header acc **78.0%**
- **Qwen**: `runs/qwen_lora/best` — job 7216508 — aggregate **0.5582**, header acc **77.0%**

---

## Table of Contents

1. [Header Accuracy Overview](#1-header-accuracy-overview)
2. [Header Confusion Patterns](#2-header-confusion-patterns)
3. [Text Length Analysis](#3-text-length-analysis)
4. [Hallucination Analysis](#4-hallucination-analysis)
5. [Metric Penalty: Semantically Correct but Score-Penalised](#5-metric-penalty-semantically-correct-but-score-penalised)
6. [Section-Specific Deep Dives](#6-section-specific-deep-dives)
7. [Shared Failure Examples](#7-shared-failure-examples)
8. [Root Cause Summary](#8-root-cause-summary)
9. [Actionable Improvements](#9-actionable-improvements)

---

## 1. Header Accuracy Overview

| Model | Correct / Total | Accuracy |
|---|---|---|
| Llama FT | 156 / 200 | **78.0%** |
| Qwen FT | 154 / 200 | **77.0%** |

Overall accuracy is nearly identical. The two models fail on *different* sections, mostly in opposite directions.

### Per-Label Accuracy

| Section | n (test) | Llama FT | Qwen FT | Δ (Qwen − Llama) |
|---|---|---|---|---|
| MEDICATIONS | 10 | **100%** | **100%** | 0 |
| PLAN | 1 | **100%** | **100%** | 0 |
| DISPOSITION | 1 | **100%** | **100%** | 0 |
| GYNHX | 1 | **100%** | **100%** | 0 |
| IMAGING | 1 | **100%** | **100%** | 0 |
| LABS | 1 | **100%** | **100%** | 0 |
| IMMUNIZATIONS | 1 | **100%** | **100%** | 0 |
| FAM/SOCHX | 45 | **98%** | 93% | −5pp |
| ALLERGY | 12 | **92%** | **92%** | 0 |
| PASTSURGICAL | 7 | **86%** | **86%** | 0 |
| GENHX | 53 | **85%** | 74% | −11pp |
| ROS | 17 | **82%** | 76% | −6pp |
| PASTMEDICALHX | 14 | **79%** | 71% | −8pp |
| CC | 11 | 36% | **73%** | **+37pp** |
| ASSESSMENT | 11 | 27% | **55%** | **+28pp** |
| EXAM | 5 | 40% | 40% | 0 |
| OTHER_HISTORY | 3 | 0% | 33% | +33pp |
| EDCOURSE | 4 | 0% | 0% | 0 |
| DIAGNOSIS | 1 | 0% | 0% | 0 |
| PROCEDURES | 1 | 0% | 0% | 0 |

**Key observation**: Llama is stronger on the narrative heavy sections (GENHX, FAM/SOCHX, ROS, PASTMEDICALHX). Qwen is stronger on the compact, structured sections (CC, ASSESSMENT). Both completely fail EDCOURSE, DIAGNOSIS, and PROCEDURES.

Both models predict the wrong header on the same **34 examples** simultaneously — these are the hardest cases in the test set and are discussed in [Section 7](#7-shared-failure-examples).

---

## 2. Header Confusion Patterns

### 2.1 Llama FT — Top Confusions (wrong only)

| Reference → Predicted | Count |
|---|---|
| CC → GENHX | 5 |
| ASSESSMENT → GENHX | 4 |
| ROS → GENHX | 2 |
| ASSESSMENT → PASTMEDICALHX | 2 |
| GENHX → FAM/SOCHX | 2 |
| OTHER_HISTORY → PASTMEDICALHX | 2 |
| GENHX → EDCOURSE | 2 |
| EDCOURSE → EXAM | 2 |

**Dominant pattern**: Llama predicts **GENHX when wrong**. Any short dialogue touching symptoms, history, or patient background triggers a full GENHX narrative expansion. ASSESSMENT examples (diagnoses discussed) are reframed as a patient history paragraph instead of a numbered list.

### 2.2 Qwen FT — Top Confusions (wrong only)

| Reference → Predicted | Count |
|---|---|
| GENHX → CC | 5 |
| FAM/SOCHX → OTHER_HISTORY | 2 |
| GENHX → FAM/SOCHX | 2 |
| EDCOURSE → ASSESSMENT | 2 |
| GENHX → EDCOURSE | 2 |
| ASSESSMENT → CC | 2 |

**Dominant pattern**: Qwen predicts **CC when wrong**. It extracts the chief complaint from what should be a full GENHX narrative and stops. ASSESSMENT examples are also compressed to a single noun phrase and mislabeled as CC.

### 2.3 The Opposite Failure on CC / GENHX

This is the most striking structural finding:

| Dialogue type | Llama error | Qwen error |
|---|---|---|
| Long symptom history (→ GENHX) | Correct GENHX narrative | Collapses to CC noun phrase |
| Brief symptom mention (→ CC) | Expands into GENHX narrative | Correct CC noun phrase |

Llama over-expands; Qwen over-compresses. When one is right the other is wrong on the same dialogue.

---

## 3. Text Length Analysis

### 3.1 Overall

| Metric | Llama FT | Qwen FT | Reference |
|---|---|---|---|
| Mean pred words | 46.5 | 32.3 | 40.6 |
| Overall pred/ref ratio | **1.31** | **0.90** | 1.0 |

Llama over-generates by 31% on average; Qwen under-generates by 10%. Llama is closer to reference length in absolute terms but its excess is concentrated in specific failure sections.

### 3.2 Per-Section Length Ratio (pred words / ref words)

| Section | n | Llama ratio | Qwen ratio | Ref mean words |
|---|---|---|---|---|
| ASSESSMENT | 11 | **4.36** | 0.59 | 25.6 |
| CC | 11 | **3.93** | 1.19 | 7.7 |
| OTHER_HISTORY | 3 | 0.87 | **7.04** | 3.0 |
| EDCOURSE | 4 | 1.69 | **0.25** | 90.0 |
| GENHX | 53 | 1.15 | 0.88 | 93.7 |
| ROS | 17 | 0.93 | 0.99 | 32.1 |
| FAM/SOCHX | 45 | 0.96 | 0.83 | 19.7 |
| EXAM | 5 | 0.74 | 1.04 | 15.0 |
| PASTMEDICALHX | 14 | 0.82 | 0.53 | 12.9 |
| ALLERGY | 12 | 0.72 | 0.50 | 4.8 |
| MEDICATIONS | 10 | 0.77 | 0.71 | 12.0 |
| PASTSURGICAL | 7 | 0.68 | 0.66 | 7.0 |

**Llama problem sections**: ASSESSMENT (4.36×) and CC (3.93×) — when the header is wrong, Llama writes a full GENHX narrative, inflating word count 4–5×. EDCOURSE (1.69×) also over-generates but for the wrong section.

**Qwen problem sections**: EDCOURSE (0.25×) produces 1–2 bullets from a 90-word narrative. PASTMEDICALHX (0.53×) strips qualifiers. OTHER_HISTORY (7.04×) — only 3 examples; one severely over-generated outlier distorts the mean.

**Sections both models handle well**: ROS (both near 1.0), EXAM (both near 1.0), GENHX (Llama 1.15, Qwen 0.88).

---

## 4. Hallucination Analysis

### 4.1 Number Hallucination

Numbers in the prediction that do not appear in either the reference or the source dialogue:

| Metric | Llama FT | Qwen FT |
|---|---|---|
| Examples with ≥1 hallucinated number | **42 / 200 (21%)** | 32 / 200 (16%) |
| Examples with ≥2 hallucinated numbers | **15 / 200 (7.5%)** | 14 / 200 (7%) |
| Mean hallucinated numbers per example | **0.31** | 0.26 |
| Both models hallucinate simultaneously | 23 / 200 (11.5%) | — |

Llama hallucinates numbers more frequently. The main sources are: fabricated patient age, fabricated pain scores, and specific lab/drug values not mentioned in the dialogue.

**Important caveat**: Many "extra numbers" are correct transcriptions of spoken numbers in the dialogue ("fifty-five" → "55", "January eighth" → "01/08"). These are not true hallucinations. The figures above represent numbers absent from *both* the dialogue and the reference.

### 4.2 The "55-Year-Old" Artifact (Llama-specific)

Llama has a severe systematic hallucination: it predicts **"55-year-old"** as the patient's age in 18 examples where neither the dialogue nor the reference mentions this age.

| Metric | Llama FT | Qwen FT |
|---|---|---|
| Pred contains "55-year-old" | 21 | 2 |
| Of those: hallucinated (not in ref or dialogue) | **18** | 0 |

This is a training artifact — age 55 occurs frequently enough in the MTS-Dialog training set that Llama uses it as a default when no age is provided. Qwen does not exhibit this pattern at all.

**Example (Llama, ASSESSMENT section)**:
> Dialogue: *"Patient: I have been having some headaches. Doctor: We'll need to evaluate that..."* (no age mentioned)
> Llama pred: *"The patient is a **55-year-old female** who comes in today for a followup appointment. She has been having some headaches..."*
> Reference: *"Migraine headache."*

The model writes a full GENHX narrative (wrong section), fabricates age 55, and ignores the ASSESSMENT format entirely.

### 4.3 Shared Hallucination Example (Both Models)

23 examples where both models introduce numbers absent from dialogue and reference. The clearest shared case:

> **Section**: CC — `idx=21`
> **Dialogue**: *"Doctor: Is this chest pain new? Patient: Yeah, last few nights. Doctor: How would you describe it? Patient: Gnawing. Doctor: How long? Patient: Few seconds. Doctor: Rate it. Patient: Moderate..."* (no age, no numeric rating given)
>
> **Ref**: `Multiple problems, main one is chest pain at night.`
>
> **Llama** (hallucinated: `55`, `5`, `6/10`): *"The patient is a **55-year-old** male... He rates it as **5 to 6/10** in intensity..."*
>
> **Qwen** (hallucinated: `67`, `5`, `6`, `10`): *"The patient is a **67-year-old** male... He rates it as **5 or 6 out of 10**..."*

Both models fabricate a specific numeric age (different values — 55 vs 67) and convert "moderate" into a numeric pain score. The reference for this CC section requires only a short noun phrase, not a patient narrative. Both models get the section format wrong and hallucinate details in the process.

---

## 5. Metric Penalty: Semantically Correct but Score-Penalised

A significant portion of low-scoring predictions are clinically acceptable but diverge from the reference in surface form, directly penalising ROUGE and BLEURT.

### 5.1 Condensed vs Contextualised Phrasing

Models produce terse clinical shorthand; references include qualifying context that ROUGE rewards:

| Section | Llama pred | Qwen pred | Reference |
|---|---|---|---|
| PASTMEDICALHX | `Diabetes, hepatitis C, and HIV.` | `Metastatic prostate cancer.` | `He has diabetes, but this is well controlled. He also has hepatitis C and HIV.` |
| FAM/SOCHX | `Grandmother with arthritis. Father with psoriasis.` | `His grandmother had arthritis. His father has psoriasis.` | `Positive for arthritis in his grandmother. No history of pediatric arthritis. There is history of psoriasis in his dad.` |
| FAM/SOCHX | `Retired. Social drinker. No tobacco or illicit drug use.` | (same) | `He is retired from the social security administration x 20 years. He travels a lot and is extremely active. He does not smoke. He consumes alcohol socially only. He does not use illicit drugs. He is married.` |

In these cases the model's answer is factually correct and clinically appropriate. It is penalised purely because the reference includes more surrounding context (negations, qualifiers, elaborations). BLEURT is especially sensitive to this since it scores semantic similarity against the reference sentence-by-sentence.

Llama has 13 examples of this pattern across FAM/SOCHX and PASTMEDICALHX; Qwen has 19.

### 5.2 ASSESSMENT Format Mismatch

The reference ASSESSMENT is inconsistently formatted in the training/test set — sometimes a numbered list, sometimes prose. Both models predict a format that may not match the reference even when the content is correct:

| Case | Llama | Qwen | Reference | Impact |
|---|---|---|---|---|
| Numbered list vs prose | `1. Anxiety. 2. Hypertension.` | `1. Anxiety. 2. Hypertension.` | `Generalized anxiety and hypertension, both under fair control.` | ROUGE −0.3 |
| Prose vs numbered list | (prose narrative) | `Hypertension, hyperlipidemia, and osteoarthritis.` | `1. Hypertension. 2. Hypercholesterolemia. 3. Osteoarthritis. 4. Fatigue.` | Missing item drops ROUGE |
| Different term level | `1. Clinical sinusitis. 2. Right otitis media with effusion.` | `1. Bilateral otitis media. 2. Sinusitis.` | `Ongoing purulent rhinitis. Probable sinusitis and serous otitis.` | Synonym mismatch |

All three predictions are reasonable clinical assessments. The score penalty is driven by surface string differences, not clinical accuracy.

### 5.3 MEDICATIONS: Name-Only vs Full Entry

Both models list drug names only; references include dose, route, timing, and context. This is the most systematic metric gap for a high-accuracy section:

| Model | Pred | Reference |
|---|---|---|
| Llama | `Amoxil and Aldex.` | `None except the Amoxil and Aldex started on Monday.` |
| Both | `[drug name].` | `[drug name] [dose] [route] [frequency], recently started / discontinued / continued.` |

Average pred/ref ratio for MEDICATIONS: Llama 0.77, Qwen 0.71. ROUGE-1 for this section is around 0.40 despite 100% header accuracy.

---

## 6. Section-Specific Deep Dives

### 6.1 GENHX (n=53 — 26.5% of test set)

The most common section. Llama is more accurate (85% vs 74%) but Qwen produces better-proportioned text when correct.

**Llama errors (8/53)**: Mostly FAM/SOCHX or EDCOURSE misclassification. When wrong, generates a long narrative in the wrong section.

**Qwen errors (14/53)**: 5 predicted as CC (extracts chief complaint only), 2 as FAM/SOCHX, 2 as EDCOURSE. When Qwen correctly identifies GENHX, text averages ~82 words vs Llama's ~103 words — Qwen is tighter but still well within clinical range.

### 6.2 ASSESSMENT (n=11)

| | Llama FT | Qwen FT |
|---|---|---|
| Header correct | 3 / 11 (27%) | 6 / 11 (55%) |
| Numbered list format used | Rarely (when header wrong, writes GENHX narrative) | Usually |
| Content when correct | Misses qualifiers ("under fair control", "slowly resolving") | Correct but terse |

Llama's main failure: writes a full GENHX narrative beginning with "The patient is a 55-year-old..." for ASSESSMENT examples. The 55-year-old artifact is most visible here — 4 of Llama's 8 wrong ASSESSMENT predictions contain this fabricated age.

Qwen is better at ASSESSMENT labelling (55%) but format inconsistency still hurts scores — numbered list vs prose reference gives near-zero ROUGE-2 even when the diagnoses are identical.

### 6.3 CC (n=11)

| | Llama FT | Qwen FT |
|---|---|---|
| Header correct | 4 / 11 (36%) | 8 / 11 (73%) |
| Length when correct (mean words) | ~8 words | ~6 words |
| Length when wrong | ~90 words (GENHX narrative) | ~5 words |

Qwen has much better CC accuracy. When Llama gets CC wrong it writes a 80–120 word GENHX paragraph (ratio 3.93×). When Qwen gets CC wrong it usually misclassifies as something else (ASSESSMENT, FAM/SOCHX), not the reverse.

### 6.4 EDCOURSE (n=4) — Complete Failure

Neither model correctly classifies any EDCOURSE example.

| | Llama FT | Qwen FT |
|---|---|---|
| Header accuracy | 0 / 4 | 0 / 4 |
| Predicted as | ASSESSMENT, EXAM, EXAM, GENHX | ASSESSMENT, EXAM, IMAGING, ASSESSMENT |
| Mean pred length | ~133 words | ~25 words |
| Mean ref length | ~90 words | ~90 words |

Llama over-generates (writes a GENHX-style narrative), Qwen under-generates (2–5 word lists or single bullets).

**Why both fail**: EDCOURSE dialogues are clinician-to-clinician or doctor-to-family exchanges describing what has already happened in the hospital. They superficially resemble GENHX (narrative), EXAM (examination findings described), and ASSESSMENT (outcomes discussed). With only 4 test examples and very few training examples, no reliable EDCOURSE label boundary is learned.

**Example** (Poison Control consult):
> Dialogue: *"Doctor: I spoke with Poison Control. They said it's a nontoxic ingestion if she even ingested it."*
> Llama → ASSESSMENT: `Nontoxic ingestion.`
> Qwen → ASSESSMENT: `1. Possible ingestion of a small amount of a nontoxic liquid.`
> Reference (EDCOURSE): `I discussed the case with Poison Control and apparently this is actually relatively small quantity and it is likely to be a nontoxic ingestion if she even ingested...`

Both models extract the clinical conclusion; neither writes the narrative hospital course.

### 6.5 MEDICATIONS (n=10)

100% header accuracy for both models. The failure is entirely in text content — both strip clinical context:

| | Llama FT | Qwen FT | Reference |
|---|---|---|---|
| Avg words | 5.9 | 6.0 | 12.0 |
| Ratio | 0.77 | 0.71 | — |

Models list drug names; references include when the drug was started, at what dose, the route, and any relevant context. This is a prompt failure, not a model capability failure — the model knows doses exist in the dialogue but does not include them without explicit instruction.

### 6.6 Rare Labels (n≤4 each: DIAGNOSIS, PROCEDURES, OTHER_HISTORY, EDCOURSE)

| Section | n | Llama acc | Qwen acc | Common misprediction |
|---|---|---|---|---|
| EDCOURSE | 4 | 0% | 0% | EXAM, ASSESSMENT |
| OTHER_HISTORY | 3 | 0% | 33% | PASTMEDICALHX |
| DIAGNOSIS | 1 | 0% | 0% | CC |
| PROCEDURES | 1 | 0% | 0% | ROS |

Insufficient training examples for any of these. The model defaults to the nearest high-frequency semantically-adjacent label. Oversampling or hardcoded few-shot examples in the prompt are the only fixes without more data.

---

## 7. Shared Failure Examples

### 7.1 Both Models Wrong Header — Top Cases

34 of 200 examples (17%) have wrong header predictions from both models simultaneously.

| idx | Ref header | Llama pred | Qwen pred | Why both fail |
|---|---|---|---|---|
| 5 | FAM/SOCHX | OTHER_HISTORY | OTHER_HISTORY | Social/family details in ambiguous context |
| 11 | ROS | GENHX | OTHER_HISTORY | BCG vaccine question looks like social history |
| 14 | GENHX | FAM/SOCHX | FAM/SOCHX | Dialogue opens with family questions before turning to HPI |
| 16 | EXAM | ROS | ROS | Symptom questions ("Is your skin turning blue?") before physical findings |
| 17 | PASTSURGICAL | PASTMEDICALHX | PASTMEDICALHX | "No health problems" framing without surgical vocabulary |

The shared errors cluster around **boundary ambiguity** — dialogues that contain elements of multiple sections or that start with one genre before pivoting to another.

### 7.2 Shared Number Hallucination — The Chest Pain Case

The most instructive shared hallucination (both models, `CC` section, `idx=21`):

> **Dialogue**: A brief exchange where the doctor asks about chest pain. Patient says it started "a few nights ago", lasts "a few seconds", rates it as "moderate." **No age mentioned. No numeric rating given.**
>
> **Reference (CC)**: `Multiple problems, main one is chest pain at night.`
>
> **Llama** hallucinated: `55`, `5`, `6/10`
> Pred: *"The patient is a **55-year-old** male… He rates it as **5 to 6/10** in intensity…"*
>
> **Qwen** hallucinated: `67`, `5`, `6`, `10`
> Pred: *"The patient is a **67-year-old** male… He rates it as **5 or 6 out of 10**…"*

Both models:
1. Predict the wrong section (GENHX-style narrative for a CC that should be 5 words)
2. Fabricate a specific patient age (different values: Llama 55, Qwen 67)
3. Convert "moderate" into a specific numeric score (5–6/10)

The age hallucination confirms that each model has a different default age learned from training — Llama defaults to 55, Qwen to 67. The numeric pain score is a plausible inference but is not grounded in the dialogue.

---

## 8. Root Cause Summary

### Llama FT

| Error | Root cause |
|---|---|
| ASSESSMENT 27% accuracy | Always writes GENHX-style prose for diagnosis-related dialogues; never learned the numbered-list format for ASSESSMENT |
| CC 36% accuracy | Short symptom dialogues trigger GENHX expansion — demographics and history added that the CC section should not contain |
| "55-year-old" in 18 hallucinated cases | Training artifact: age 55 is frequent in training data; model defaults to it when no age is specified |
| EDCOURSE / rare labels 0% | Insufficient training examples; model defaults to EXAM, GENHX, or ASSESSMENT |
| MEDICATIONS truncation (0.77×) | Treats MEDICATIONS as name enumeration; drops dose/route/timing context |
| Over-generation on ASSESSMENT/CC (4×) | When header is wrong, the GENHX-style narrative inflates word count dramatically |

### Qwen FT

| Error | Root cause |
|---|---|
| GENHX 74% accuracy (5 misses → CC) | Strong CC-format calibration over-generalises: terse clinical phrasing is applied even to dialogues that need a full paragraph |
| EDCOURSE 0%, severe truncation (0.25×) | Rare label + compression bias: extracts the headline finding and stops |
| FAM/SOCHX → OTHER_HISTORY confusion (2×) | Combined family+social dialogues with ambiguous context are filed under OTHER_HISTORY |
| ASSESSMENT format inconsistency | Uses numbered list where ref is prose; scores badly on ROUGE-2 even when diagnoses are correct |
| MEDICATIONS truncation (0.71×) | Same name-only enumeration failure as Llama |
| PASTMEDICALHX strips qualifiers (0.53×) | Drops "well controlled", "as previously described", negation context — content correct, BLEURT penalised |

### Shared Failures (both models)

| Error | Root cause |
|---|---|
| 34 shared header errors (17%) | Genuine label ambiguity — dialogues spanning multiple sections; neither model resolves the boundary correctly |
| EDCOURSE 0% | Clinician-to-clinician style mimics GENHX and ASSESSMENT; insufficient training examples for reliable boundary |
| MEDICATIONS at 0.71–0.77× | Both list names only; prompt does not require dose/route/context to be preserved |
| Number hallucination (11.5% both) | Models infer plausible numeric values (age, scores) from dialogue context rather than copying them verbatim |
| Default age anchoring | Llama defaults to 55, Qwen to 67 — each has a training-data-driven prior age when the dialogue is silent |
| Condensed vs contextualised phrasing | Both produce valid clinical shorthand that ROUGE penalises vs the fuller reference phrasing |

---

## 9. Actionable Improvements

### Prompt-Level (no retraining)

**1. Explicit MEDICATIONS instruction**
The section prompt already says "include drug name, dose, route, and frequency." This may need to be moved to the user turn as a reminder specific to MEDICATIONS examples, since both models still ignore it.

**2. Hardcoded rare-label examples**
EDCOURSE, DIAGNOSIS, PROCEDURES, and OTHER_HISTORY collectively contribute 9 test examples at ~0% accuracy. Adding one hardcoded few-shot example per label directly in the system prompt (not via RAG, which is stochastic) would give reliable format anchors.

**3. Suppress the "55-year-old" default (Llama)**
Add to the system prompt: *"Do not infer or assume the patient's age if it is not stated in the dialogue."* This would prevent the training artifact from contaminating predictions.

**4. ASSESSMENT format reinforcement**
Add to the ASSESSMENT section rule: *"Use a numbered list only if the reference dialogue presents multiple distinct conditions. For a single condition, write a brief phrase."* This would reduce the format mismatch penalty on examples where the reference uses prose.

### Training-Level (requires retraining)

**5. Oversample rare labels**
Duplicate or augment training examples for EDCOURSE, LABS, PROCEDURES, DIAGNOSIS, and OTHER_HISTORY to bring them closer to the frequency of GENHX and FAM/SOCHX. Even 10–20 extra examples of each would likely move these sections from 0% to meaningful accuracy.

**6. MEDICATIONS gold targets with full context**
In the current training data, some MEDICATIONS reference summaries are also name-only. Augmenting these with fuller gold targets (dose + route + timing extracted from the dialogue) would teach the model through supervision.

**7. ASSESSMENT gold target normalisation**
The training set has inconsistent ASSESSMENT targets — some numbered, some prose. Normalising them to a consistent numbered-list format would remove the format inconsistency the models are learning.

### Post-hoc Rule-Based Fixes

**8. GENHX/CC length heuristic**
If predicted label is CC and the generated text is longer than 15 words → reclassify as GENHX. This would catch Llama's 5 CC→GENHX errors where it writes a full narrative.

**9. ASSESSMENT format enforcer**
If predicted label is ASSESSMENT and the generated text contains "The patient is a" → the model has written a GENHX narrative; re-run generation with an explicit bullet-list instruction.

**10. Age suppression post-processing (Llama)**
Strip "The patient is a 55-year-old" prefix from predictions where "55" does not appear in the source dialogue. This is a targeted fix for the most common hallucination artifact.
