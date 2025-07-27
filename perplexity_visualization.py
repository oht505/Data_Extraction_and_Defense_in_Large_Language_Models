import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

sns.set(style="whitegrid")

file_list = [
    "extraction_Nano_124_264.log",
    "extraction_dpNano_124_Nano.log",
    "extraction_264_124_Nano.log",
    "extraction_loraOnly_124_264_Nano(N=10000).log"
]

model_labels = {
    "extraction_Nano_124_264.log": ("NanoGPT", "Non-DP"),
    "extraction_dpNano_124_Nano.log": ("NanoGPT", "DP"),
    "extraction_264_124_Nano.log": ("GPT2-Large", "Non-DP"),
    "extraction_loraOnly_124_264_Nano(N=10000).log": ("GPT2-Large", "DP")
}

all_samples = []

def parse_samples_from_file(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    samples = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^\d+: (Nano|PPL-[\w]+)=.*', line):
            match = re.match(r'^(\d+): (?:Nano|PPL-[\w]+)=([\d.]+), Zlib=([\d.]+), score=([-\d.]+)', line)
            if match:
                sample_id = int(match.group(1))
                nano_ppl = float(match.group(2))
                zlib = float(match.group(3))
                score = float(match.group(4))
            else:
                i += 1
                continue

            i += 1
            text_lines = []
            while i < len(lines) and not re.match(r'^\d+: (Nano|PPL-[\w]+)=.*', lines[i]):
                text_lines.append(lines[i].strip(" '\n"))
                i += 1

            full_text = ' '.join(text_lines)

            model_name, dp_status = model_labels[filename]
            samples.append({
                'id': sample_id,
                'nano_ppl': nano_ppl,
                'zlib_entropy': zlib,
                'score': score,
                'text': full_text,
                'model': model_name,
                'dp': dp_status
            })
        else:
            i += 1

    return samples

# Collect and Parse
for file in file_list:
    if os.path.exists(file):
        all_samples.extend(parse_samples_from_file(file))

df_all = pd.DataFrame(all_samples)
df_all_clean = df_all.dropna(subset=["nano_ppl", "zlib_entropy"])


# Plot 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)

model_order = ['NanoGPT', 'GPT2-Large']
dp_order = ['Non-DP', 'DP']

for i, model in enumerate(model_order):
    for j, dp_status in enumerate(dp_order):
        ax = axes[i, j]
        subset = df_all_clean[(df_all_clean['model'] == model) & (df_all_clean['dp'] == dp_status)]

        # For NanoGPT - apply different range
        if model == "NanoGPT" and dp_status == "DP":
            subset = subset[subset['nano_ppl'] < 10000] 
        else:
            subset = subset[subset['nano_ppl'] < 1000]   

        sns.scatterplot(data=subset, x='nano_ppl', y='zlib_entropy', ax=ax, alpha=0.7, s=30)
        ax.set_title(f"{model} ({dp_status})")
        ax.set_xlabel("Perplexity")
        ax.set_ylabel("Zlib Entropy")

plt.tight_layout()
# plt.suptitle("Perplexity vs. Zlib Entropy Across Models and DP Settings", y=1.02)
plt.show()
