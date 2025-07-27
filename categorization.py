import re
import csv
import time
import spacy
from collections import defaultdict, Counter
from datetime import datetime
import os
import gc

nlp = spacy.load("en_core_web_sm")

CATEGORY_EXTRACTORS = {
    "US and International News": lambda doc, text: re.findall(r"(?i)\b(CNN|BBC|Reuters|The New York Times|The Guardian|AP News|Bloomberg|Biden|Trump|White House|UN|NATO|Ukraine|China|Russia)\b", text),
    "Log files and error reports": lambda doc, text: re.findall(r"(Traceback \(most recent call last\)|Exception|Error|\.log|\.trace)", text),
    "License, terms of use, copyright notices": lambda doc, text: [m.group() for m in re.finditer(r"(?i)(license|copyright|terms of use|all rights reserved)", text)],
    "Lists of named items (games, countries, etc.)": lambda doc, text: re.findall(r"(?:[A-Z][a-z]+,\s*){2,}[A-Z][a-z]+", text),
    "Forum or Wiki entry": lambda doc, text: re.findall(r"(\bUser talk:|\bWikipedia:|\bTalk:|\b[Ww]iki\b|\bFAQ\b)", text),
    "Valid URLs": lambda doc, text: re.findall(r"(?i)\b((?:https?|ftp|www)[\.:/\\]?\w+(?:[\.:/\\]?\w+)*\.(?:com|net|org|edu|gov|io|co|info|biz|us|uk|de|jp|kr))", text),
    "Named individuals (non-news samples only)": lambda doc, text: [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
    "Promotional content (products, subscriptions, etc.)": lambda doc, text: re.findall(r"(?i)\b(buy now|limited offer|subscribe|click here|free trial|flash sale|redeem now|% off|exclusive offer)\b", text),
    "High entropy (UUIDs, base64 data)": lambda doc, text: re.findall(r"[A-Za-z0-9+/=]{30,}", text),
    "Contact info (address, email, phone, twitter, etc.)": lambda doc, text: re.findall(
        r"(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-zA-Z]{2,})"  # email
        r"|(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}"  # phone
        r"|@\w{1,15}"  # twitter handle
        r"|\b\d{1,5}\s+(?:[A-Z][a-z]+\s)+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Dr|Drive|Court|Ct)\b",
        text
    ),
    "Code": lambda doc, text: re.findall(r"(?:def |class |function\s+|</?[a-z]+>)", text),
    "Configuration files": lambda doc, text: re.findall(r"(?:\[.*?\]|^[a-zA-Z0-9_]+\s*=\s*.+$)", text, re.MULTILINE),
    "Religious texts": lambda doc, text: re.findall(r"\b(Genesis|Bible|Quran|Torah|Psalm|Jesus|Allah|God|prophet)\b", text, re.IGNORECASE),
    "Pseudonyms": lambda doc, text: re.findall(r"(?:aka|also known as|anno)\s+[A-Z][a-z]+", text),
    "Donald Trump tweets and quotes": lambda doc, text: re.findall(r"(?i)(Donald Trump:|@realDonaldTrump|Make America Great Again|fake news|donald trump)", text),
    "Web forms (menu items, instructions, etc.)": lambda doc, text: re.findall(r"<form.+?</form>|input type=", text, re.DOTALL),
    "Tech News": lambda doc, text: re.findall(r"(?i)\b(TechCrunch|The Verge|Wired|Gizmodo|Ars Technica|Engadget|developer conference|software release|gadget review|AI|robotics|cloud computing|startup)\b", text),
    "Lists of numbers (dates, sequences, etc.)": lambda doc, text: re.findall(r"(?:\d+[\s,])+\d+", text),
}

# Input log file 
sample_file = "Empty input cases/extraction_loraOnly_124_264_Nano(N=1000000).log"
sample_name = os.path.splitext(os.path.basename(sample_file))[0]
resume_index_filename = f"resume_index_{sample_name}.txt"
summary_output_file = f"final_summary_{sample_name}.csv"
progress_csv_file = f"progress_temp_{sample_name}.csv"

# Read samples from the results of perplexity
with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
    sample_raw = f.read()

# Read dataset for comparison
with open("merged_all.txt", 'r', encoding='utf-8') as f:
    log_data = f.read()

# Pre-processing samples
sample_blocks = re.findall(r"\('(.+?)'\)", sample_raw, re.DOTALL)
sample_blocks = [" ".join(block.replace("'", "").replace("\n", " ").split()) for block in sample_blocks]

for i, sample in enumerate(sample_blocks):
    print(f"idx {i}: {sample}\n")

resume_index = 0
category_to_matches = defaultdict(list)
category_to_count = Counter()

# Load resume index if exists
if os.path.exists(resume_index_filename):
    with open(resume_index_filename, "r") as f:
        resume_index = int(f.read().strip())

# Load prior match data if progress file exists
if os.path.exists(progress_csv_file):
    with open(progress_csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["Category"]
            count = int(row["Count"])
            examples = row["Examples"].split('; ') if row["Examples"] else []
            category_to_count[category] = count
            category_to_matches[category] = examples

print(f"Starting from sample {resume_index + 1} / {len(sample_blocks)}")

# Categorize from resume index to last sample block index
for i in range(resume_index, len(sample_blocks)):
    text = sample_blocks[i]

    print(f"[{datetime.now().isoformat()}] ▶ Sample {i+1} / {len(sample_blocks)}")

    with nlp.select_pipes(disable=["parser", "tagger"]):
        doc = nlp(text)

    for category, extractor in CATEGORY_EXTRACTORS.items():
        infos = extractor(doc, text)
        for info in set(infos):
            if info in log_data:
                category_to_matches[category].append(info)
                category_to_count[category] += 1

    # Index file
    with open(resume_index_filename, "w") as f:
        f.write(str(i + 1))

    # Intermediate .csv file
    with open(progress_csv_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Count", "Examples"])
        for category in CATEGORY_EXTRACTORS.keys():
            matches = category_to_matches.get(category, [])
            writer.writerow([
                category,
                category_to_count.get(category, 0),
                "; ".join(sorted(set(matches)))
            ])

    gc.collect()

# Write final summary
with open(summary_output_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Count", "Examples"])
    for category in CATEGORY_EXTRACTORS.keys():
        matches = category_to_matches.get(category, [])
        writer.writerow([
            category,
            category_to_count.get(category, 0),
            "; ".join(sorted(matches))
        ])
    writer.writerow([])
    writer.writerow(["Total Samples", len(sample_blocks)])

# Remove temp files
if os.path.exists(progress_csv_file):
    os.remove(progress_csv_file)
if os.path.exists(resume_index_filename):
    os.remove(resume_index_filename)

print("\nSummary CSV complete:", summary_output_file)
