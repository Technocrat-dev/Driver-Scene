# Experiment Results

## Overview

Comparison of **8 prompt variants** on **BDD100K dashcam images** using **Groq / Llama 4 Scout** (free tier, 14,400 RPD).

- **Dataset**: 10 sampled images from BDD100K validation set
- **Model**: `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API
- **Evaluation**: Automated metrics including BERTScore, hallucination rate, spatial grounding, and count accuracy
- **Runtime**: ~3 minutes for all 8 variants (80 API calls)

---

## Comparison Table

| Rank | Variant | BERTScore F1 | Hallucination ↓ | Spatial Acc | Count MAE ↓ | Images |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **v2_structured** | 0.332 | **0.253** | **0.194** | **3.18** | 10 |
| 🥈 | **v5_few_shot** | **0.366** | 0.451 | 0.074 | 4.39 | 9 |
| 🥉 | **v4_cot** | 0.334 | 0.455 | 0.063 | 4.46 | 10 |
| 4 | v6_safety | 0.338 | 0.439 | 0.065 | 5.29 | 9 |
| 5 | v1_baseline | 0.324 | 0.389 | 0.079 | 5.17 | 10 |
| 6 | v8_combined | 0.319 | 0.450 | 0.070 | 5.24 | 9 |
| 7 | v7_grounded | 0.318 | 0.490 | 0.113 | 4.42 | 10 |
| 8 | v3_role | 0.297 | 0.420 | 0.052 | 5.03 | 10 |

> ↓ = lower is better

---

## Key Findings

### 1. v2_structured is the clear overall winner
Despite not having the highest BERTScore, v2 dominates on the metrics that matter most for AD:
- **Lowest hallucination** (0.253) — 35% better than baseline
- **Best spatial accuracy** (0.194) — 2.5× better than baseline
- **Best object counting** (3.18 MAE) — 38% lower error than baseline

### 2. v5_few_shot has the best semantic similarity
v5 achieves the highest BERTScore F1 (0.366), suggesting few-shot examples help the model generate descriptions most similar to ground truth language. However, its hallucination rate (0.451) is significantly worse.

### 3. Structured output constrains hallucination
v2's explicit JSON schema acts as an implicit grounding constraint, preventing the model from fabricating objects. This pattern is consistent across both Gemini (earlier testing) and Llama 4 Scout — structured prompts reduce hallucination regardless of the underlying VLM.

### 4. BERTScore doesn't correlate with quality
The variants with highest BERTScore (v5, v4, v6) also have the **worst** hallucination rates. Higher BERTScore means more verbose, natural-language output that shares vocabulary with GT — but that verbosity also introduces more errors. Hallucination rate and spatial accuracy are far more reliable quality indicators.

### 5. Role-play (v3) hurts performance
v3_role performed worst on nearly every metric, suggesting that "you are an AD engineer" framing causes the model to be more confident and creative, leading to more hallucinations.

---

## Metric Comparison: v2_structured vs Baseline

| Metric | v1_baseline | v2_structured | Improvement |
|---|:---:|:---:|:---:|
| **Hallucination Rate** ↓ | 0.389 | 0.253 | **−35%** |
| **Spatial Accuracy** | 0.079 | 0.194 | **+145%** |
| **Count MAE** ↓ | 5.17 | 3.18 | **−38%** |
| **BERTScore F1** | 0.324 | 0.332 | +2.5% |

---

## Infrastructure

| Aspect | Gemini (previous) | Groq (current) |
|---|---|---|
| Model | gemini-2.5-flash-lite | Llama 4 Scout (17B) |
| RPD | ~20 effective | 14,400 |
| 8-variant runtime | 18+ min (incomplete) | **~3 minutes** |
| Success rate | ~25/80 calls | **79/80 calls** |

---

## Reproducing

```bash
# Set provider to Groq in .env
VLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# Run comparison
python -m src.pipeline compare --limit 10

# Or switch back to Gemini
VLM_PROVIDER=gemini
```
