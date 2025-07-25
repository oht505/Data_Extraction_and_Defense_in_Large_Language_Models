# Data Extraction and Defense in Large Language Models 

This repository contains codes for **memorization** and **privacy risks** in large language models (LLMs), particularly focusing on potential leakage of **personally identifiable information (PII)**. We replicate and extend **Extracting Training Data from Large Languagae Models [here](https://github.com/ftramer/LM_Memorization)** and propose a **Differentially Private Stochastic Gradient Descent (DP-SGD)** as a defense, optimized for both:

- NanoGPT (trained from scratch)
- GPT-2 Large (fine-tuned with LoRA adapters)

**Full Paper**: [`AI539_NLP_with_DL_Final_Report_Data_Extraction_and_Defense.pdf`](./AI539_NLP_with_DL_Final_Report_Data_Extraction_and_Defense.pdf)

# What We DO

1. **Train and fine-tune LLMs** on datasets containing sensitive data (e.g. Enron emails, synthetic PII)
2. **Evaluate privacy risk** via black-box data extraction (Carlini-style) by generating and filtering outputs.
3. **Defend** against memorization with a scalable, memory-efficient version of **DP-SGD**

---

# Installation

1. Install dependencies
``` 
pip install -r requirements.txt
```

2. Recommended hardware
- NVIDIA H100 GPU
- At least **40 GB of GPU memory** is required to train GPT-2 Large with DP-SGD

---



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

## Team & Contributions

- **Hyuntaek Oh (me)** - Train NanoGPT from scratch, Extraction pipeline, Memorization analysis, Categorization
- **Woonki Kim** - Data Preprocessing, Differential privacy defense implementation
- **Prayoga** - Data Preprocessing, Train GPT2-Large, Memorization analysis

Oregon State University  
{ohhyun, kimwoon, prayoga}@oregonstate.edu

---
