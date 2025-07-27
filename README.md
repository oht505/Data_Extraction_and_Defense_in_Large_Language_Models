# Data Extraction and Defense in Large Language Models 

This repository contains codes for **memorization** and **privacy risks** in large language models (LLMs), particularly focusing on potential leakage of **personally identifiable information (PII)**. We replicate and extend **Extracting Training Data from Large Languagae Models ([here](https://github.com/ftramer/LM_Memorization))** and propose a **Differentially Private Stochastic Gradient Descent (DP-SGD)** as a defense, optimized for both:

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

# Preprocessing

The data cleaning and merging steps were developed by my teammate. For full details and implementation, please refer to his repository:
([Preprocssing Guide by Woonki](https://github.com/woonki94/privacy-defense-gpt2))  

---

# Train language models 

### NanoGPT (Train from Scratch)

The NanoGPT training pipeline was implemented by me using custom configuration based on the original [Karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) repository. 

For a complete training guide and setup, please refer to the documentation maintained by my teammate (using the exact code I contributed):
[NanoGPT Training Guide by Woonki](https://github.com/woonki94/privacy-defense-gpt2)

> Note: The code in the above guide was originally implemented by me and Later reused in the shared team repository.

### GPT-2 Large (LoRA Fine-Tuning)

This part was handled entirely by teammates, including DP-SGD implementation and LoRA fine-tuning.
You can find the full training pipeline here:
[GPT-2 Large Training Guide by Woonki](https://github.com/woonki94/privacy-defense-gpt2)

---

# Run Extraction



---

# Run Categorization (Optional)



---

# Measure Accuracy

The implementation of memory-efficient DP-SGD (including ghost clipping and virtual batching) was handled by a team member:
You can find the full training scripts here:
([link](https://github.com/woonki94/privacy-defense-gpt2))

---

## Key Results

### 1. DP-SGD Loss & Privacy Budget Progression

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/172daf13-fabd-44cb-aeb9-27cb60ac2942" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/fcb64885-2dfb-4cdc-a91d-d21a55a81691" width="500"/></td>
  </tr>
</table>

> The graphs show how the privacy budget (ε) grows over training steps and how loss decreases for both NanoGPT and GPT-2 Large.

---

### 2. Verbatim Memorization Examples

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/f2907295-941c-4b42-8f44-cdc83e7c9d2b" width="500"/><br/>(a) Extracted output from NanoGPT</td>
    <td><img src="https://github.com/user-attachments/assets/4273c224-b463-4469-b8b8-37ab3d144d69" width="500"/><br/>(b) Matched string "bill.rust@enron.com" in training dataset</td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/822d25e2-421b-46ab-a0df-2f8c97464e83" width="500"/><br/>(c) Extracted output from NanoGPT</td>
    <td><img src="https://github.com/user-attachments/assets/6227688a-fd18-40d6-8068-4aa48f5ad9b0" width="500"/><br/>(d) Matched string "I Am ... Sasha Fierce" in training dataset</td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/cbcd021b-ee8d-4011-8732-553e72c1ae18" width="500"/><br/>(e) Extracted output from NanoGPT</td>
    <td><img src="https://github.com/user-attachments/assets/c190cbb6-92d1-46b3-9323-b044b47a2eac" width="500"/><br/>(f) Matched string "getty images donald trump" in the training dataset</td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/231171c6-9c63-4499-90c4-99b3c1f5e0f5" width="500"/><br/>(g) Extracted output from GPT-2 Large</td>
    <td><img src="https://github.com/user-attachments/assets/43b83e6c-e8ad-4853-8b03-faea2d29d830" width="500"/><br/>(h) Matched string "steamapps\workshop\content" in training dataset</td>
  </tr>
</table>

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
