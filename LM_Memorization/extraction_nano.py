"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel, PeftConfig

# For NanoGPT
from model import GPTConfig, GPT
import tiktoken

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)

enc = tiktoken.get_encoding("gpt2")

def encode(s):
    return enc.encode(s, allowed_special={"<|endoftext|>"})

def decode(toks):
    return enc.decode(toks)

def load_nanoGPT_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_args = ckpt['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = ckpt['model']
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    return model

def load_gpt2_model(path):
    config = PeftConfig.from_pretrained(path)
    base_model = GPT2LMHeadModel.from_pretrained(config.base_model_name_or_path).to(device)
    model = PeftModel.from_pretrained(base_model, path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def calculatePerplexity_nano(sentence, model):
    input_ids = torch.tensor(encode(sentence)).unsqueeze(0).to(device)
    if input_ids.size(1) < 2:  # Skip too short sentence
        return torch.tensor(float("inf"))

    targets = input_ids[:, 1:].contiguous().to(device)
    inputs = input_ids[:, :-1].contiguous().to(device)
    with torch.no_grad():
        logits, loss = model(inputs, targets)
    return torch.exp(loss)

def calculatePerplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=100):
    idxs = np.argsort(metric)[::-1][:n]
    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, score={metric[idx]:.3f}")
        print()
        pprint(samples[idx])
        print("\n")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--N', type=int, default=5000, help="Number of samples to generate")
    # parser.add_argument('--batch-size', type=int, default=50, help="Batch size for generation")
    parser.add_argument('--N', type=int, default=100000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=100, help="Batch size for generation")
    return parser.parse_args(argv)

def main():
    print(f"using device: {device}")
    args = parse_arguments(sys.argv[1:])

    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40

    print("Loading models...")
    nano_model = load_nanoGPT_model("ckpt.pt")
    gpt2_124_model, gpt2_124_tokenizer = load_gpt2_model("checkpoint-1240000")
    gpt2_264_model, gpt2_264_tokenizer = load_gpt2_model("checkpoint-264000")
    nano_model.eval()
    gpt2_124_model.eval()
    gpt2_264_model.eval()

    samples = []
    scores = {"124": [], "264": [], "Nano": [], "Lower": [], "zlib": []}

    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            # Take first N tokens as prompt
            prompts = ["<|endoftext|>"] * args.batch_size
            # input_len = 1
            # inputs = gpt2_124_tokenizer(prompts, return_tensors='pt', padding=True)
            # inputs = {k: v.to(device) for k, v in inputs.items()}

            # output_sequences = gpt2_124_model.generate(
            #     input_ids=inputs['input_ids'].to(device),
            #     attention_mask=inputs['attention_mask'].to(device),
            #     max_length=input_len + seq_len,
            #     do_sample=True,
            #     top_k=top_k,
            #     top_p=1.0,
            #     temperature=1.0,
            #     repetition_penalty=1.2
            # )

            # texts = gpt2_124_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            
            # For NanoGPT
            encoded_prompts = [torch.tensor(encode(p), dtype=torch.long, device=device)[None, ...] for p in prompts]
   
            input_batch = torch.cat(encoded_prompts, dim=0)
            output_batch = nano_model.generate(
                idx=input_batch,
                max_new_tokens=seq_len,
                temperature=1.0,
                top_k=top_k,
                repetition_penalty=1.2
            )
            texts = [decode(out[input_batch.shape[1]:].tolist()) for out in output_batch]

            for text in texts:
                # perplexity of GPT2-L and NanoGPT
                ppl_gpt2_124 = calculatePerplexity(text, gpt2_124_model, gpt2_124_tokenizer)
                ppl_gpt2_264 = calculatePerplexity(text, gpt2_264_model, gpt2_264_tokenizer)
                ppl_nano = calculatePerplexity_nano(text, nano_model)

                # perplexity on lower-case sample
                p_lower = calculatePerplexity(text.lower(), gpt2_124_model, gpt2_124_tokenizer)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores["124"].append(ppl_gpt2_124)
                scores["Nano"].append(ppl_nano)
                scores["264"].append(ppl_gpt2_264)
                scores["Lower"].append(p_lower)
                scores["zlib"].append(zlib_entropy)

            pbar.update(args.batch_size)

    scores["124"] = np.asarray([x.cpu().item() for x in scores["124"]])
    scores["264"] = np.asarray([x.cpu().item() for x in scores["264"]])
    scores["Nano"] = np.asarray([x.cpu().item() for x in scores["Nano"]])
    scores["Lower"] = np.asarray([x.cpu().item() for x in scores["Lower"]])
    scores["zlib"] = np.asarray(scores["zlib"])

    # Sort by perplexity
    metric = -np.log(scores["Nano"])
    print(f"======== top sample by Nano perplexity: ========")
    print_best(metric, samples, "Nano", scores["Nano"])
    print()
    print()

    # Sort by ratio of log perplexities of 124 and Nano models
    metric = np.log(scores["124"]) / np.log(scores["Nano"])
    print(f"======== top sample by ratio of L-124 and Nano perplexities: ========")
    print_best(metric, samples, "Nano", scores["Nano"], "PPL-124", scores["124"])
    print()
    print()

    # Sort by ratio of log perplexities of 264 and Nano models
    metric = np.log(scores["264"]) / np.log(scores["Nano"])
    print(f"======== top sample by ratio of Nano and L-264 perplexities: ========")
    print_best(metric, samples, "PPL-264", scores["264"], "Nano", scores["Nano"])
    print()
    print()

    # Sort by ratio of log perplexities of lower-case and normal-case perplexities
    metric = np.log(scores["Lower"]) / np.log(scores["Nano"])
    print(f"======== top sample by ratio of lower-case and normal-case perplexities: ========")
    print_best(metric, samples, "Nano", scores["Nano"], "PPL-L-Lower", scores["Lower"])
    print()
    print()

    # Sort by ratio of Zlib entropy and L perplexity
    metric = scores["zlib"] / np.log(scores["Nano"])
    print(f"======== top sample by ratio of Zlib entropy and Nano perplexity: ========")
    print_best(metric, samples, "Nano", scores["Nano"], "Zlib", scores["zlib"])


if __name__ == '__main__':
    main()
