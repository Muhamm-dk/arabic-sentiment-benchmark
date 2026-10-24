"""Fine-tune an encoder for binary sentiment, tracked in MLflow.

Run: uv run python -m arabic_sentiment.finetune --dataset ajgt --model marbert --seed 0
Model selection uses validation macro-F1 per epoch; test is scored once with the best epoch.
Outputs: results/runs/<run>.json, results/preds/<run>.parquet, MLflow run in mlflow.db
"""
import argparse
import json
import math
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from arabic_sentiment.baseline import bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "marbert": "UBC-NLP/MARBERT",
    "xlmr": "FacebookAI/xlm-roberta-base",
}
MAX_LEN = {"labr": 256, "hard": 256, "hard_notitle": 256, "astd": 96, "ajgt": 96}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len)


def batches(enc, labels, idx, bs, tok):
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        feats = [{k: enc[k][i] for k in enc.keys()} for i in b]
        x = tok.pad(feats, return_tensors="pt")
        yield {k: v.cuda() for k, v in x.items()}, torch.tensor(labels[b], device="cuda")


@torch.no_grad()
def predict_proba(model, enc, labels, tok, bs=64):
    model.eval()
    out = []
    for x, _ in batches(enc, labels, np.arange(len(labels)), bs, tok):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**x).logits.float().softmax(-1).cpu())
    return torch.cat(out).numpy()


def macro_f1(y, probs):
    return float(f1_score(y, probs.argmax(-1), average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MAX_LEN))
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--train_subsample", type=int, default=None,
                    help="fixed seeded subsample of the training split (data seed 42, independent of --seed)")
    ap.add_argument("--run_tag", default=None, help="suffix for the run name, keeps result files separate")
    ap.add_argument("--save", action="store_true", help="save best weights to outputs/<run>")
    a = ap.parse_args()

    set_seed(a.seed)
    max_len = a.max_len or MAX_LEN[a.dataset]
    run = f"{a.dataset}-{a.model}-s{a.seed}"
    df = pd.read_parquet(ROOT / f"data/processed/{a.dataset}.parquet")
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if a.train_subsample and a.train_subsample < len(parts["train"]):
        # same row positions for hard and hard_notitle (identical rows and order), so only the text differs
        parts["train"] = parts["train"].sample(a.train_subsample, random_state=42).reset_index(drop=True)
        run += f"-n{a.train_subsample}"
    if a.run_tag:
        run += f"-{a.run_tag}"

    name = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).cuda()
    enc = {s: encode(tok, d.text, max_len) for s, d in parts.items()}
    y = {s: d.label.to_numpy() for s, d in parts.items()}

    n_batches = math.ceil(len(y["train"]) / a.batch_size)
    total_steps = a.epochs * math.ceil(n_batches / a.grad_accum)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("arabic-sentiment-benchmark")
    with mlflow.start_run(run_name=run):
        mlflow.log_params({**vars(a), "hf_model": name, "max_len_used": max_len,
                           "n_train": len(y["train"]), "n_val": len(y["val"]), "n_test": len(y["test"])})
        best_f1, best_epoch, best_state = -1.0, -1, None
        rng = np.random.default_rng(a.seed)
        for epoch in range(1, a.epochs + 1):
            model.train()
            order = rng.permutation(len(y["train"]))
            losses = []
            for i, (x, t) in enumerate(batches(enc["train"], y["train"], order, a.batch_size, tok)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**x).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), t)
                (loss / a.grad_accum).backward()
                losses.append(loss.item())
                if (i + 1) % a.grad_accum == 0 or i == n_batches - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            val_f1 = macro_f1(y["val"], predict_proba(model, enc["val"], y["val"], tok))
            print(f"{run} epoch {epoch}: train_loss={np.mean(losses):.4f} val_macro_f1={val_f1:.4f}")
            mlflow.log_metrics({"train_loss": float(np.mean(losses)), "val_macro_f1": val_f1}, step=epoch)
            if val_f1 > best_f1:
                best_f1, best_epoch = val_f1, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs = predict_proba(model, enc["test"], y["test"], tok)
        pred = probs.argmax(-1)
        result = {
            "run": run, "dataset": a.dataset, "model": a.model, "seed": a.seed,
            "best_epoch": best_epoch, "val_macro_f1": round(best_f1, 4),
            "test_macro_f1": round(float(f1_score(y["test"], pred, average="macro")), 4),
            "test_macro_f1_ci95": bootstrap_ci(y["test"], pred),
            "test_accuracy": round(float(accuracy_score(y["test"], pred)), 4),
            "n_test": int(len(pred)),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        print(json.dumps(result, indent=2))
        mlflow.log_metrics({k: result[k] for k in ("test_macro_f1", "test_accuracy", "peak_vram_gib")})
        out = parts["test"].assign(pred=pred, prob_pos=probs[:, 1])
        out.to_parquet(ROOT / f"results/preds/{run}.parquet", index=False)
        (ROOT / f"results/runs/{run}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(ROOT / f"results/runs/{run}.json"))
        if a.save:
            d = ROOT / f"outputs/{run}"
            model.save_pretrained(d)
            tok.save_pretrained(d)


if __name__ == "__main__":
    main()
