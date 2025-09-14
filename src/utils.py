import os
import re
import yaml
import torch
import random
import logging
import numpy as np
from glob import glob
from collections import OrderedDict

logging.getLogger()

# -------------------------------
# File & YAML utilities
# -------------------------------

def load_file(filename):
    """Read a text file where sentences are separated by newlines."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def parse_yaml(filepath):
    """Read & parse a YAML file."""
    with open(filepath, "r", encoding='utf-8') as fin:
        return yaml.safe_load(fin)

def write_yaml(data, filepath):
    """Write a dictionary into a YAML file."""
    with open(filepath, 'w', encoding='utf-8') as fout:
        yaml.dump(data, fout, default_flow_style=False, allow_unicode=True)

# -------------------------------
# Checkpoint utilities
# -------------------------------

def load_checkpoint(ckpt_path, device, option="best"):
    """
    Load a model checkpoint for inference or continued training.
    """
    stat_dict = None
    if option == "best":
        best_path = os.path.join(ckpt_path, "best.ckpt")
        if os.path.exists(best_path):
            stat_dict = torch.load(best_path, map_location=device)
        else:
            logging.warning("Best checkpoint not found. Falling back to latest.")
            option = "latest"

    if option == "latest":
        checkpoints = glob(f"{ckpt_path}/*.ckpt")
        if os.path.exists(f"{ckpt_path}/best.ckpt"):
            checkpoints.remove(f"{ckpt_path}/best.ckpt")
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_path}")
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))[-1]
        logging.info(f"Loading latest checkpoint: {latest_checkpoint}")
        stat_dict = torch.load(latest_checkpoint, map_location=device)

    # Remove unused keys if needed
    if stat_dict is None:
        return None
    ignore_keys = {"bn.weight", "bn.bias", "bn.running_mean",
                   "bn.running_var", "bn.num_batches_tracked", "fc.weight", "fc.bias"}
    new_stat_dict = {k.partition('.')[-1]: v for k, v in stat_dict.items() if k.partition('.')[-1] not in ignore_keys}
    return OrderedDict(new_stat_dict)

# -------------------------------
# Punctuation and case utilities
# -------------------------------

def punc_map(label):
    """Map punctuation class label to symbol."""
    mapping = {
        "COMMA": ",",
        "PERIOD": ".",
        "QUESTION": "?",
        "EXCLAMATION": "!",
        "COLON": ":",
        "SEMICOLON": ";",
        "O": ""
    }
    return mapping.get(label, "")

def get_case(word):
    """Detect the case of a given word."""
    if word.isupper():
        return 'A'  # ALL_CAPS
    elif word.istitle():
        return 'F'  # FIRST_CAP
    else:
        return 'O'  # OTHER

def apply_labels_to_input(tokens_count_per_sent, total_tokens, punc_preds, class_to_punc):
    """
    Reconstruct sentences from tokens + predictions.
    - Keeps last subword punctuation.
    - Ignores [PAD] tokens.
    - Merges subwords starting with ##.
    """
    labeled_sentences = []
    idx = 0

    for sent_len in tokens_count_per_sent:
        sent_tokens = []
        tokens_in_sentence = 0

        while tokens_in_sentence < sent_len and idx < len(total_tokens):
            token = total_tokens[idx]
            if token == "[PAD]":
                idx += 1
                continue

            # Merge subwords
            while idx + 1 < len(total_tokens) and total_tokens[idx + 1].startswith("##"):
                token += total_tokens[idx + 1][2:]
                idx += 1

            # Apply punctuation if not "O"
            punc_label = class_to_punc[punc_preds[idx]]
            if punc_label != "O":
                token += punc_map(punc_label)

            sent_tokens.append(token)
            tokens_in_sentence += 1
            idx += 1

        labeled_sentences.append(" ".join(sent_tokens))

    return labeled_sentences

# -------------------------------
# Misc utilities
# -------------------------------

def sum_params(model):
    """Sum the weights/parameters of a given model."""
    return sum(np.sum(p.cpu().data.numpy()) for p in model.parameters())

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sort_alphanumeric(lst):
    """Sort a list in human order."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)
