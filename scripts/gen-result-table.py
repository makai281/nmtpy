#!/usr/bin/env python
import os
import sys

from tabulate import tabulate

def parse_log_file(f):
    opt_keys = ["alpha_c", "batch_size", "clip_c",
                "data", "decay_c", "dicts",
                "dropout", "embedding_dim",
                "optimizer", "rnn_dim",
                "shuffle", "sort", "valid_freq",
                "valid_metric"]
    opt_dict = {}
    with open(f, 'r') as fh:
        for line in fh:
            if " -> " in line:
                # Options
                opt = line.split(" ", 3)[-1].strip()
                k, v = opt.split(" -> ")
                if k in opt_keys:
                    opt_dict[k] = v
            elif "vocabulary size" in line:
                _type, rest = line.strip().split(" ", 3)[2:]
                size = int(rest.split(":")[-1])
                if _type == "Source":
                    opt_dict['src_vocab'] = size
                elif _type == "Target":
                    opt_dict['trg_vocab'] = size
            elif "data: " in line:
                _, _, _type, _, size, _ = line.strip().split(" ", 5)
                if _type == "Training":
                    opt_dict['train_size'] = int(size)
                elif _type == "Validation":
                    opt_dict['valid_size'] = int(size)
            elif "Best " in line:
                parts = line.strip().replace("[", "").replace("]", "").split(" ")
                _parts = []
                for p in parts:
                    if p not in ("", "Validation", "="):
                        _parts.append(p)
                parts = _parts
                val_key = parts[4].lower()
                opt_dict[val_key] = "%.3f (%d)" % (float(parts[5].strip(",")), int(parts[2]))
    return opt_dict

if __name__ == '__main__':
    results = []
    colnames = ["src vocab", "trg vocab", "train", "valid", "emb_dim", "rnn_dim", "BLEU", "METEOR", "loss"]
    cols = ["src_vocab", "trg_vocab", "train_size", "valid_size", "embedding_dim", "rnn_dim", "bleu", "meteor", "loss"]
    results.append(["id"] + colnames)
    files = sorted(sys.argv[1:])
    for idx, logf in enumerate(files, 1):
        result = parse_log_file(logf)
        result = [result.get(col, "-") for col in cols]
        results.append([idx] + result)

    # Sort by BLEU
    bleu_id = colnames.index("BLEU") + 1
    results[1:] = sorted(results[1:], key=lambda x: x[bleu_id], reverse=True)
    # Make BLEU bold
    results[1][bleu_id] = "**%s**" % results[1][bleu_id]

    files = [[r[0], files[r[0]-1]] for r in results[1:]]

    print tabulate(results, headers="firstrow", tablefmt="pipe")
    print
    print tabulate(files, headers=["id", "file"], tablefmt="pipe")
