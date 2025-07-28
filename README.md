# Data Extraction and Defense in Large Language Models 

This repository contains codes for **memorization** and **privacy risks** in large language models (LLMs), particularly focusing on potential leakage of **personally identifiable information (PII)**. We replicate and extend **Extracting Training Data from Large Languagae Models ([ftramer/LM_Memorization](https://github.com/ftramer/LM_Memorization))** and propose a **Differentially Private Stochastic Gradient Descent (DP-SGD)** as a defense, optimized for both:

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

This step evaluates how much **memorized content** each model reproduces through generation. We generate text samples from the model and measure how similar they are to the original dataset. Then, we compare different models (NanoGPT / GPT-2 Large, with or without DP) using **perplexity and compression** metrics.

### Preparation

Make sure the following are ready:
- Model checkpoints:
```bash
extraction_LMs.py
chkpt/
├── large/
│   ├── plain/checkpoint-245000/...
│   └── dp_sgd/checkpoint-245000/...
└── nano/
    ├── plain/checkpoint-60000.pt
    └── dp-sgd/checkpoint-60000.pt
```
- 'tiktoken', 'transformers', 'peft', 'torch', 'zlib', etc. (see 'requirements.txt')
- (Optional) For real-prompt sampling: Common Crawl WET file (see [Here for wet file](https://github.com/ftramer/LM_Memorization))
- (Optional) To make replication easier, pretrained checkpoints for all models are provided by my teammate (see [Pretrained Checkpoints](https://github.com/woonki94/privacy-defense-gpt2))


### Run the Script

The easiest way to run the extraction process is simply:

```bash
python extraction_LMs.py
```

This command runs the script using **default settings**, which are sufficient for most use cases. By default, it generates 100,000 samples using the gpt2_dp model and prints the top 100 based on perplexity. The extraction script 'extraction_LMs.py' supports arguments to customize the generation and evaluation process. All argements have default values, so the script works out of the box. But for advanced control, here's what each option does:

| Argument             | Type   | Default   | Description |
|----------------------|--------|-----------|-------------|
| `--gen-model`        | str    | `gpt2_dp` | Selects the model to use. Options: `nano`, `nano_dp`, `gpt2`, `gpt2_dp`. |
| `--batch-size`       | int    | `100`     | Number of samples generated per batch. Larger batch = faster but more memory usage. |
| `--N`                | int    | `10000`   | Total number of samples to generate. |
| `--num-print`        | int    | `100`     | Number of top samples (by score) to print and save. |
| `--seq-len`          | int    | `256`     | Maximum number of tokens to generate for each sample. |
| `--top-k`            | int    | `40`      | Top-k sampling filter: only consider the top-K tokens at each step. |
| `--internet-sampling`| flag   | `False`   | If set, uses real prompts from Common Crawl. Otherwise, uses empty prompts. |
| `--wet-file`         | str    | `None`    | Path to the Common Crawl `.wet` file (required only if `--internet-sampling` is set). |

Here is an example of using options:
```bash
python extraction_LMs.py --gen-model gpt2_dp --N 10000 --num-print 20 --internet-sampling --wet-file commoncrawl.warc.wet
```

After running the script, the following output will be generated:
- 'results_model.txt':
  A text file containing the top N generated samples, sorted by perplexity or metric score.
  
- Console output:
  The samples will also be printed to terminal, including their perplexity and metric score.


**Example snippet from 'results_gpt2_dp.txt':**
![gpt2_dp screenshot](https://github.com/user-attachments/assets/6f4cc3ca-294c-4652-92cb-48a2972ce414)


You can then feed this file into the categorization script to analyze memorized content.

> Tip: If you run the script multiple times with the same model, outputs will overwrite the previous 'results_model.txt' file unless renamed.


---

# Run Categorization

### Categorize Extracted Samples

Once you've generated the model outputs, this step helps **identify and count memorized content** by category (e.g., names, emails, URLs, etc.). Make sure you have a generated file from the previous extraction step like "results_gpt2_dp.txt". Place it in the root directory or specify its path when running the script. 

```bash
python categorization.py --sample-file results_nano.txt
```

This scripts scans each sample and checks for matches against predefined categories using rule-based matchers and named entity recognition. It also generates a final summary CSV with matched counts and examples:
```
final_summary_results_nano.csv
```

This CSV continas:

| Category                                                | Count   | Examples                                      |
|---------------------------------------------------------|---------|-----------------------------------------------|
| `News`                                                  | 145     | Russia; Ukraine; biden; bloomberg; china; ... |
| `License, terms of use, copyright notices`              | 3       | copyright; license                            |
| `Valid URLs`                                            | 0       |                                               |
| `Named individuals (non-news samples only)`             | 351     | Aiden; Amy; Andy; Arlene; Bass; Bermuda;  ... |
| `Promotional content (products, subscriptions, etc.)`   | 66      | subscribe; ...                                |
| `Contact info (address, email, phone, twitter, etc.)`   | 27      | 518-454-5387; `amy.fitzpatrick@enron.com`; ...  |
| `Code`                                                  | 40      | `[10]`; `[1]`; `[cri]`; `[ep]`; ...           |
| `Configuration files`                                   | 0       |                                               |
| `Religious texts`                                       | 13      | Jesus; bible; god; jesus; psalm; ...          |
| `Donald Trump tweets and quotes`                        | 0       |                                               |
|                          |     |    |
| `Total Category Matches` | 645 |    |
| `Total Samples`          | 500 |    |

> You can apply this script to any output file from `extraction_LMs.py`, including those from `nano`, `gpt2`, or `dp` variants.
> Intermediate files (for resuming or debugging) are deleted automatically after successful run.

### Visualization 

You can now use this CSV to visualize category-level memorization differences between models.

(Under construction...)

---

# Measure Accuracy

The implementation of memory-efficient DP-SGD (including ghost clipping and virtual batching) was handled by a team member:
You can find the full training scripts here:
([Measure Accuracy Guide by Woonki](https://github.com/woonki94/privacy-defense-gpt2))

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
