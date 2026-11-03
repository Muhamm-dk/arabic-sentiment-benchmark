# Chapter 9: The fine-tuning pipeline

## What

One script, `src/arabic_sentiment/finetune.py`, trains any of the three encoders on any dataset and writes everything needed to compare runs: scores, test predictions and an MLflow record. Every encoder number in this guide came from it.

Run: `uv run python -m arabic_sentiment.finetune --dataset ajgt --model marbert --seed 0`

## Why one plain PyTorch loop

- Every step is visible and can be explained line by line in an interview: padding, mixed precision, the learning-rate schedule, gradient accumulation, clipping and epoch selection.
- The same code trains all three models on all datasets, so differences between results come from the model and the data, not from different code paths.
- It avoids the `Trainer` class, whose behaviour is configured through many arguments and changes between versions. I did not compare the two on this task.

Cost: the loop has no resume, no checkpoints per epoch, and no CPU fallback (`.cuda()` is hard-coded).

## The run contract

Run name: `{dataset}-{model}-s{seed}`, plus `-n{N}` when the training set is subsampled, plus `-{run_tag}` when a tag is given.

| Run | Name |
|---|---|
| AJGT, MARBERT, seed 1 | `ajgt-marbert-s1` |
| HARD, AraBERT, 20,000-row subsample | `hard-arabert-s0-n20000` |
| XLM-R tuned configuration | `ajgt-xlmr-s0-tuned` |
| Ablation (MARBERT at 128 tokens) | `labr-marbert-s0-len128` |

Outputs, all keyed by the run name:
- `results/runs/<run>.json`: scores (macro-F1, interval, accuracy, best epoch, peak VRAM).
- `results/preds/<run>.parquet`: test rows with `pred` and `prob_pos` (not tracked in git; it contains dataset texts).
- an MLflow run in `mlflow.db` (not tracked either).
- with `--save`, the model and tokenizer in `outputs/<run>`.

Consumers (`compare.py`, `dialect_analysis.py`, `report.py`, `combined_report.py`) accept only `{dataset}-{model}-s{seed}`, optionally `-n{N}`, and `-tuned` for the XLM-R tuned runs. An ablation must use another tag and will be ignored by them. The rule exists because of the incident in "What went wrong".

## How: the code

### Encoding and batching

```python
def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len)

def batches(enc, labels, idx, bs, tok):
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        feats = [{k: enc[k][i] for k in enc.keys()} for i in b]
        x = tok.pad(feats, return_tensors="pt")
        yield {k: v.cuda() for k, v in x.items()}, torch.tensor(labels[b], device="cuda")
```

- Texts are tokenized once up front, truncated at `max_len` (256 for reviews and the combined set, 96 for tweets) and not padded.
- `batches` takes row indices, builds one batch at a time and pads it to the longest item in that batch (dynamic padding). Most reviews are short, so most batches are much shorter than 256. The peak VRAM comes from the batch that contains the longest text.
- `tok.pad` works for BERT-style inputs (`input_ids`, `token_type_ids`, `attention_mask`) and for XLM-R (no `token_type_ids`), because it uses whatever keys the encoding has.
- Test and validation text is encoded before training starts but is not used until the end of each epoch (validation) and the last step (test).

### Model

```python
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).cuda()
```

The checkpoint's encoder is loaded in fp32 and a new 2-class head is created (the `MISSING` keys in the load report, Chapter 8). `.cuda()` moves everything to the GPU.

### Optimizer and schedule

```python
n_batches = math.ceil(len(y["train"]) / a.batch_size)
total_steps = a.epochs * math.ceil(n_batches / a.grad_accum)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
```

- **AdamW** keeps two running averages per parameter and applies weight decay separately from the gradient update (the "W"). It decays all parameters, including biases and layer-norm weights. The common recipe excludes those (memory), and the effect of the difference was not tested.
- **Schedule:** the learning rate rises linearly from 0 to its peak over the first 10% of optimizer steps (warmup), then falls linearly to 0 at the last step. One schedule spans all epochs. So a later epoch is evaluated at a lower learning rate than an earlier one, and the best epoch is not necessarily the last.
- Why warmup (explanation from memory): the new head is random and the optimizer's running averages are unreliable at the start, so full-size steps early can damage the pretrained weights.

Worked numbers from the code formulas (arithmetic):

| Run | Train rows | Batches per epoch | Epochs | Optimizer steps | Warmup steps |
|---|---|---|---|---|---|
| AJGT, batch 32 | 1,440 | 45 | 5 | 225 | 22 |
| ASTD, batch 32 | 1,935 | 61 | 5 | 305 | 30 |
| LABR, batch 32 | 8,301 | 260 | 4 | 1,040 | 104 |
| HARD (20,000 subsample), batch 32 | 20,000 | 625 | 3 | 1,875 | 187 |
| LABR, XLM-R, batch 16 with accumulation 2 | 8,301 | 519 | 4 | 1,040 | 104 |
| Combined model, batch 32 | 31,674 | 990 | 3 | 2,970 | 297 |

### One training step

```python
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
```

- **Mixed precision.** Weights stay in fp32. Inside `autocast`, PyTorch runs the heavy operations (matrix multiplications) in bf16 and chooses the precision of others itself. bf16 has the same number range as fp32 but fewer digits, so training needs no loss scaling. fp16 has a narrower range and needs a gradient scaler. The cost is lower numeric precision, and I did not compare against fp32 training.
- **Loss.** The model is called without labels, and the loss is computed explicitly: `cross_entropy` on the two scores, cast to float first. Plain cross-entropy with no class weights: classes are balanced in AJGT and LABR, and macro-F1 is the metric. ASTD is 68% negative, and no weighting or resampling was used. The effect was not tested.
- **Gradient accumulation.** `loss / grad_accum` makes the summed gradients equal to the gradient of the mean loss over the larger effective batch. XLM-R at batch 16 with accumulation 2 is therefore an effective batch of 32, which keeps its optimizer settings equal to the other models. If the last group of batches is incomplete, the final step is slightly smaller than the others. The effect is small and was not measured.
- **Gradient clipping.** If the total gradient norm exceeds 1.0, all gradients are scaled down to that norm. It guards against occasional large gradients and was not tuned.
- **`set_to_none=True`** frees gradient memory between steps.
- `loss.item()` copies a number to the CPU each step. It costs a little speed and gives the epoch's mean training loss.

### Epoch selection and the test

```python
val_f1 = macro_f1(y["val"], predict_proba(model, enc["val"], y["val"], tok))
if val_f1 > best_f1:
    best_f1, best_epoch = val_f1, epoch
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
...
model.load_state_dict(best_state)
probs = predict_proba(model, enc["test"], y["test"], tok)
```

- After each epoch the model is scored on validation. When it improves, a full copy of the weights is saved in CPU memory. The strict `>` keeps the earlier epoch on a tie.
- After the last epoch the best weights are restored, and test is scored once. The test split is not touched before that line.
- `predict_proba` runs in eval mode (dropout off), without gradients, in bf16, in batches of 64, in the original row order, which keeps predictions aligned with the saved rows.
- Weakness: validation on tiny sets picks noisy epochs (AraBERT on ASTD seed 2 picked epoch 1 and scored 0.8306 on test, Chapter 6).

### Seeds, subsample, outputs

```python
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
...
rng = np.random.default_rng(seed)   # batch order, one permutation per epoch
...
parts["train"] = parts["train"].sample(a.train_subsample, random_state=42)   # fixed, independent of --seed
```

- The seed controls head initialization, dropout and batch order. The training subsample uses a fixed seed (42), so every model and every seed sees the same 20,000 HARD rows, and `hard` and `hard_notitle` pick the same row positions (checked in Chapter 4).
- Bit-for-bit reproducibility on the GPU was not tested.
- `peak_vram_gib` comes from `torch.cuda.max_memory_allocated()`. That counts tensors in use and excludes the allocator's cached memory and the CUDA context, so true usage was higher than printed. The runs fit.

### MLflow

```python
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("arabic-sentiment-benchmark")
with mlflow.start_run(run_name=run):
    mlflow.log_params({**vars(a), "hf_model": name, ...})
    mlflow.log_metrics({"train_loss": ..., "val_macro_f1": ...}, step=epoch)
    mlflow.log_metrics({"test_macro_f1": ..., "test_accuracy": ..., "peak_vram_gib": ...})
    mlflow.log_artifact(".../results/runs/<run>.json")
```

Parameters, per-epoch curves and final metrics are queryable with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`. `mlflow.db` is not in git, so the durable, reproducible record in the repo is the JSON files. The MLflow history exists only on the machine that ran the experiments.

## Hyperparameters, and why

| Setting | Value | Reason |
|---|---|---|
| Learning rate | 2e-5 (tuned XLM-R: 3e-5) | Typical for BERT-style fine-tuning (the BERT paper recommends values between 2e-5 and 5e-5, memory). Not searched |
| Batch size | 32 (XLM-R on reviews: 16 with accumulation 2) | Standard, and it fits VRAM |
| Epochs | 5 on AJGT and ASTD, 4 on LABR, 3 on HARD and the combined model, 10 for tuned XLM-R | Set by dataset size and cost. The smoke test used 5 |
| Max length | 96 tweets, 256 reviews | Covers about 99.9% of tweets and about 95% of LABR reviews for the Arabic models (Chapters 3 and 8) |
| Warmup, decay | 10%, linear | Common default |
| Weight decay | 0.01 | Common default |
| Clipping | 1.0 | Common default |
| Precision | bf16 autocast | Range of fp32, no scaler |

No hyperparameter search was run. Every encoder got the same settings, with one fixed alternative configuration for XLM-R. A search could raise every score, especially on the larger datasets.

## Alternatives

None of these were tried.

| Choice | Alternative | Note |
|---|---|---|
| Plain loop | Hugging Face `Trainer` | Less code, more configuration surface |
| Full fine-tuning | LoRA or other parameter-efficient methods | Much less optimizer memory, which would help XLM-R large. Often slightly lower accuracy |
| AdamW in fp32 | 8-bit optimizers, gradient checkpointing | Would let larger models fit |
| Best epoch on validation | Fixed epochs, early stopping, averaging the last checkpoints | Averaging would reduce epoch-selection noise |
| Plain cross-entropy | Class weights, label smoothing, focal loss | Might help ASTD's imbalance |
| Pooled first-token vector | Mean pooling over tokens | See the pooler question in Chapter 8 |
| Random batches | Length-bucketed batches | Faster training because of less padding |
| One schedule over all epochs | Restarts, cosine | Minor |
| Single learning rate | Lower rates for lower layers | Sometimes helps |

## What went wrong

| Incident | Detail |
|---|---|
| Patch applied twice | The `--train_subsample` patch was pasted twice, so the argument and the sampling block were duplicated (the second pass would have resampled and added `-n20000` twice). Fix: `git checkout` of the file, then a patch that asserts each anchor exists exactly once |
| Result files overwritten | The tuned XLM-R runs needed their own names, so `--run_tag` was added |
| Silent contamination by a tagged run | The ablation `labr-marbert-s0-len128` matched the file patterns of four results scripts, which would have averaged it into the main LABR MARBERT score. Fix: exact name patterns. No committed number was affected |
| Out of memory risk | XLM-R on reviews at batch 32 would not fit. It runs at batch 16 with accumulation 2, and its peak was 6.0 GiB against about 6.5 usable |
| `.cuda()` hard-coded | The script cannot run without a GPU |
| Wasted epochs | Several runs peaked early (AraBERT on LABR at epoch 1 of 4) |

## Numbers

Peak VRAM (`max_memory_allocated`), measured:

| Run type | AraBERT | MARBERT | XLM-R |
|---|---|---|---|
| Tweets (length 96, batch 32; XLM-R tuned batch 16 also 5.2) | 2.90 to 2.91 | 3.21 | 5.23 to 5.25 |
| LABR (length 256; XLM-R batch 16 with accumulation 2) | 4.87 | 5.18 | 5.98 |
| HARD (length 256) | 4.86 to 4.87 | 5.17 | 6.0 |
| MARBERT, LABR at length 128 | | 3.61 | |
| Combined model, MARBERT | | 5.18 | |

The fixed part is parameters times 16 bytes (Chapter 8): 2.01, 2.43 and 4.14 GiB. The rest is activations and overhead.

## Not tested

- Different learning rates, batch sizes or epochs.
- fp32 against bf16.
- Weight decay on or off for biases and layer norms.
- Class weights on ASTD.
- Bit-for-bit reproducibility on the GPU.
- Training time per run was not recorded, so no timing is claimed here.

## Check yourself

1. Why is `loss / grad_accum` used, and what effective batch does XLM-R get on reviews?
2. Why does dynamic padding make the peak VRAM depend on the longest batch?
3. What happens to the learning rate across epochs, and why is the best epoch not always the last?
4. Where in the loop is the guarantee that test data cannot influence which epoch is chosen?
5. Why does bf16 not need a gradient scaler, and what does it cost?
6. Why does the HARD subsample use a fixed seed instead of `--seed`?
7. `mlflow.db` is not in git. What does the repo keep as the durable record of each run?
8. A new ablation run is named `labr-marbert-s0-len128`. Why does it not change the main table now, and why would it have before?
