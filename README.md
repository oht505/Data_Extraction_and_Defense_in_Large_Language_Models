# OSU_AI539_NLP-with-DL_Final-Project

# Data Extraction and Defense in Large Language Models (GPT2-L & NanoGPT) 

This project investigates the memorization of training data in large language models (LLMs), particularly focusing on the extraction of personally identifiable information (PII) and the effectiveness of Differential Privacy in mitigating it.

# Project Summary

We explored whether LLMs memorize sensitive training data and evaluated **Differentially Private Stochastic Gradient Descent (DP-SGD)** as a defense.

- Models: GPT-2 Large, NanoGPT
- Tasks: Reproduction of sensitive data, Privacy evaluation
- Defense: DP-SGD, LoRA + Gradient Clipping

**Full Paper**: [`AI539_NLP_with_DL_Final_Report_Data_Extraction_and_Defense.pdf`](./AI539_NLP_with_DL_Final_Report_Data_Extraction_and_Defense.pdf)

---

## Key Results

### 1. DP-SGD Loss & Privacy Budget Progression

![appendix1](https://github.com/user-attachments/assets/f8878bfe-95a3-4536-b131-14135e0471d8)

> The graphs show how the privacy budget (ε) grows over training steps and how loss decreases for both NanoGPT and GPT-2 Large.

---

### 2. Verbatim Memorization Examples

![appendix2](https://github.com/user-attachments/assets/c4b505f2-d2f9-421c-a3db-900626e2d0cd)

> These examples demonstrate text output from models that exactly matches training data, including email addresses and named individuals.

---

### 3. Perplexity vs Entropy (for extracted samples)

![appendix3](https://github.com/user-attachments/assets/7ecffb81-5b00-4eb8-b260-60ff62315b3c)

> Higher perplexity tends to correspond to higher entropy; DP-SGD reduces both, leading to less identifiable text outputs.

---

### 4. Barplot of Sensitive Categories

![appendix4](https://github.com/user-attachments/assets/1509d1fa-5302-4735-a3d8-146f63387d7e)

> DP-SGD significantly lowers extraction of sensitive information like PII and named entities.

---

## Repository Contents

- `categorization/` – Code for categorizing memorized text samples
- `LM_Memorization/` – Codebase for extracting and analyzing memorized samples
- `AI539_NLP_with_DL_Final_Report_Data_Extraction_and_Defense.pdf` – Full report

---

## Authors

- Prayoga
- Hyun Taek Oh
- Woonki Kim  
Oregon State University  
{prayoga, ohhyun, kimwoon}@oregonstate.edu

---
