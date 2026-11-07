# Interview notes, part 2: concepts

Each answer is short enough to say in 30 seconds, with the tie to this project.

## Machine learning basics

**What are train, validation and test sets for?**
Train fits the parameters. Validation chooses epochs, settings and prompts. Test is scored once for an honest estimate. Here the epoch was chosen on validation macro-F1 and the test split was not touched until the last step of each run.

**What is overfitting, and where did you see it?**
The model fits training noise and generalizes worse. MARBERT's AJGT training loss fell from 0.466 to 0.021 over five epochs while validation macro-F1 peaked at epoch 3 (0.9278) and dropped to 0.9165. In 9 of 18 first-stage runs the best epoch was 1 or 2.

**Define precision, recall and F1.**
Precision is TP/(TP+FP), recall is TP/(TP+FN), F1 is their harmonic mean, which equals 2TP/(2TP+FP+FN). The harmonic mean is low when either is low.

**Macro, micro and weighted F1?**
Macro is the unweighted mean of per-class F1. Micro pools all counts (for single-label classification it equals accuracy). Weighted averages by class size. I used macro so both classes count equally.

**Why not accuracy?**
Imbalance. ASTD is 68% negative. A model that always says negative gets 68% accuracy and macro-F1 of about 0.40.

**What does the confusion matrix show for ASTD?**
MARBERT's 21 errors are 9 false positives and 12 false negatives. Positive is the minority (32%), so it misses about 16% of positive tweets and wrongly flags about 5% of negative ones. I used plain cross-entropy, which is a candidate cause. Not tested.

**ROC-AUC against F1?**
AUC measures ranking quality across all thresholds. F1 is for one threshold. I did not use AUC. It would suit the threshold question for the API.

**How do you handle class imbalance?**
Class weights, resampling, a moved decision threshold, or a metric that does not hide it. I used macro-F1 only and did not reweight. I would test class weights on ASTD first.

**Bias and variance here.**
TF-IDF with a linear model has higher bias and, being convex, gives the same answer every run. Fine-tuned encoders have lower bias and vary by seed (standard deviation about 0.01, AraBERT on ASTD 0.032).

**What is regularization?**
A penalty or constraint that reduces overfitting. In logistic regression C is the inverse of the L2 strength. In AdamW, weight decay 0.01. BERT also uses dropout (default 0.1, from memory).

**Why not cross-validation?**
It gives lower-noise estimates on tiny sets but multiplies every run, which was impossible for the LLM evaluations. I used fixed splits, three seeds on the small sets and bootstrap intervals.

**What is a bootstrap confidence interval?**
Resample the test rows with replacement 1,000 times, recompute the metric each time, and take the 2.5th and 97.5th percentiles. It measures noise from which test rows were drawn. It does not measure training-seed noise.

**What is a paired bootstrap, and why is it better?**
Resample the same rows for both systems and bootstrap the difference. It is narrower because rows that are hard for one system tend to be hard for the other. MARBERT against AraBERT on LABR: paired half-width about 1.3 points, against about 1.8 for two separate intervals.

**Statistical against practical significance?**
A tiny gap can be significant with 10,000 rows (AraBERT over XLM-R on raw HARD, +0.4 points) and unimportant. Also I looked at 30 intervals without correcting for multiple comparisons, so a borderline one could be chance.

**What kinds of data leakage exist?**
Duplicates across splits (LABR: 65 texts in train and test). A feature that reveals the label (HARD's leading score phrase). Group leakage (reviews of one hotel in train and test). Using test data to choose settings.

**How did you make the work reproducible?**
Fixed seeds, a lock file (`uv.lock`), fixed splits and subsamples, results as JSON in git, MLflow tracking. Limits: GPU bit-for-bit reproducibility was not tested, and the Docker image is not pinned.

## Classical text models

**How does logistic regression work?**
A linear score (weights times features plus a bias) is passed through a sigmoid to give a probability. It is trained to minimize log-loss, with L2 regularization. The problem is convex, so the solver (liblinear) finds the same answer each run.

**What is TF-IDF?**
Term frequency times inverse document frequency. Rare terms count more. Mine used sublinear tf (1 + ln tf), smoothed idf, and L2-normalized rows. Minimum document frequency 2, at most 300,000 features.

**Why character n-grams for Arabic?**
Prefixes, suffixes and dialect spellings fragment word tokens into many rare forms. Character n-grams (2 to 5 inside word boundaries) overlap between variants. They won on every dataset: 0.2 to 6.2 points on validation, biggest on tweets.

**Why tune on validation and not on test?**
Choosing the best of 12 configurations on test would make the test score optimistic. Even on validation, picking the best of 12 is a little optimistic: ASTD validation 0.8190 against test 0.7708.

## Transformers and fine-tuning

**What is subword tokenization?**
A fixed vocabulary of pieces, so any string can be written, with rare words split. WordPiece is used by AraBERT and MARBERT, SentencePiece by XLM-R. XLM-R needs 39 to 54% more tokens per word than MARBERT on our data.

**What is an embedding?**
A learned table that maps each token id to a vector (768 numbers in these models). Position embeddings are added.

**Explain self-attention.**
Each token builds a new vector as a weighted mix of all tokens. Weights come from comparing queries and keys: softmax(QKᵀ/√d)V. Bidirectional in an encoder, so each token sees both sides.

**Why divide by the square root of d?**
Dot products grow with the dimension, which pushes softmax into saturation with tiny gradients. Scaling keeps the scores in a stable range.

**Why multiple heads?**
Different heads attend to different relations. Here 12 heads of 64 dimensions each.

**What is in a transformer layer?**
Attention, a two-layer feed-forward network (768 to 3072 to 768, from memory), each wrapped in a residual connection and layer normalization. Twelve layers. Attention memory grows with the square of sequence length.

**Why do models need positions?**
Attention itself ignores order. These models add learned position embeddings up to 512.

**What is BERT-style pretraining?**
Masked language modeling: hide a share of tokens (15% in the original, from memory) and predict them from context. MARBERT dropped next-sentence prediction because tweets are short.

**Encoder against decoder models?**
Encoders (BERT) read the whole input in both directions and suit classification. Decoders (GPT, the LLMs here) generate text left to right.

**Fine-tuning, feature extraction or prompting?**
Fine-tuning updates all weights with labels. Feature extraction freezes the model and trains a head. Prompting trains nothing. I used fine-tuning and few-shot prompting and compared them.

**Why does fine-tuning work with 1,440 examples?**
The model already learned language structure in pretraining. MARBERT reached 0.9277 on AJGT against 0.8722 for TF-IDF, which learns from those 1,440 rows alone.

**Why AdamW, warmup and linear decay?**
AdamW keeps adaptive per-parameter step sizes and applies weight decay correctly. Warmup avoids damaging the pretrained weights with large early steps while the new head is random (the usual explanation, not tested). Decay settles training. Settings: lr 2e-5, 10% warmup, weight decay 0.01.

**Batch size and gradient accumulation?**
Accumulation sums gradients over several small batches before one step. XLM-R used batch 16 with accumulation 2, the same effective batch of 32, because it did not fit at 32.

**Why bf16 and not fp16?**
bf16 has the number range of fp32 with fewer digits, so no loss scaling is needed. Weights stay fp32, with autocast for the matrix multiplications. I did not compare with fp32 training.

**What is gradient clipping?**
Scale gradients down if their total norm exceeds 1.0. A guard against rare large gradients. Not tuned.

**Why is a base model about 135M to 278M parameters, and what does it cost in memory?**
Most parameters are the embedding table: 250,002 x 768 = 192M of XLM-R's 278M. AdamW in fp32 holds four numbers per parameter (weight, gradient, two states), 16 bytes: 4.14 GiB for XLM-R, 2.43 for MARBERT, 2.01 for AraBERT, plus activations. XLM-R large would need 8.3 GiB, so it did not fit.

**What are LoRA, quantization and distillation?**
LoRA trains small low-rank adapters, which cuts optimizer memory. Quantization stores weights in fewer bits for smaller, faster inference. Distillation trains a small model to imitate a large one. I used none. I would try int8 for CPU latency.

**What is calibration?**
Whether a stated confidence matches how often the model is right. Expected calibration error is the row-weighted gap between confidence and accuracy over bins. The single-dataset MARBERT scored 0.102 on LABR, the combined 0.074. Temperature scaling on validation would help. Not done.

**Softmax and cross-entropy?**
Softmax turns two scores into probabilities. Cross-entropy is the negative log probability of the correct class.

## LLMs

**What is few-shot prompting?**
Labeled examples go in the prompt and the model continues the pattern, with no weight updates. Mine: 4 positive and 4 negative training examples, an instruction, a one-word answer.

**What does temperature do?**
It scales randomness in sampling. At 0, the most likely token is picked. Hosted models can still vary slightly between days.

**Why did reasoning models need token budgets?**
Hidden reasoning tokens count against the output limit. gpt-oss with a 5-token cap returned empty text with the words in a `reasoning` field. At 1,000 tokens, 3.3% of validation replies and 7 to 10% of test replies on the tweet sets were still empty or invalid.

**Why did the LLMs not beat the encoders?**
Evidence here, not a general claim: three models, one prompt, eight shots. MARBERT, fine-tuned on thousands of labeled rows, scored above every LLM on four of five sets. Likely contributors: label conventions the model has to learn (ASTD stance labels), one prompt, and invalid replies.

**What are the risks in evaluating LLMs?**
Prompt and shot-order sensitivity, hosted models changing, quota-limited sample sizes, and contamination: public datasets may be in the training data. The encoders could not exploit that in the same way.

**Prompting, fine-tuning or RAG?**
RAG adds retrieved documents to the prompt, for knowledge. Fine-tuning teaches a task or style. Prompting is cheapest to start. Sentiment is a task, so fine-tuning fits, while RAG would not help.

**How do you handle invalid model output?**
Parse strictly, count unparseable replies as wrong, report the rate. Retrying at temperature 0 gives the same answer.

## Arabic NLP

**What is diglossia?**
Formal MSA for writing alongside regional dialects for speech, increasingly written online. A model sees both, and dialect spelling is not standardized.

**Which orthographic problems matter?**
Optional diacritics, hamza variants (أ إ آ), alef maqsura against yeh (ى ي), tatweel, repeated letters ("جميييل"). My normalizer removes diacritics and tatweel, maps hamza forms to plain alef, ى to ي, and caps repeats at two. It does not map ة to ه, because that merges different words. It had no consistent effect on the baseline.

**Why is Arabic morphologically hard?**
Prefixes and suffixes attach to words (و, ب, ال, ها), so one word has many surface forms. That favors subword or character models.

**What are dialect identification and Arabizi?**
Dialect identification labels a text's variety. Arabizi is Arabic written in Latin letters and numerals. Neither was handled in training. Arabizi was not evaluated.

**Which Arabic models exist?**
AraBERT, ARBERT and MARBERT (UBC), CAMeLBERT, QARiB, XLM-R as the multilingual baseline. The Gulf-oriented LLMs are Jais (UAE, from memory) and ALLaM (Saudi, from memory). Check current model names before the interview.

End of part 2.
