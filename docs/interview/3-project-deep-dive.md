# Interview notes, part 3: the project in depth

## Data

**Why these four datasets?**
They are the standard Arabic sentiment benchmarks: the AraBERT paper evaluates on HARD, ASTD, LABR and AJGT (plus another), and MARBERT's paper includes all four. They give two domains, formal and dialect text, and sizes from 1,800 to about 100,000. They were named in the brief, so I justified them afterwards. The gap is Gulf and Maghrebi data.

**What would you add?**
ASAD (about 100,000 tweets, three classes, Khaleeji, Hijazi, Egyptian and MSA) or AraSenTi-Tweet (Saudi). I did not use them: three-class labels, not in the brief, availability unchecked.

**Why a binary label?**
One task across all datasets keeps scores comparable. AJGT has no neutral class and 3-star reviews are ambiguous. Cost: neutral text is forced to a polarity.

**How did you map stars?**
1 and 2 negative, 4 and 5 positive, 3 dropped. LABR's hub label is an index, not the star value: index = stars minus 1.

**Which copies did you use, and what did you find?**
Hub copies for LABR, ASTD and AJGT, the GitHub zip for HARD. The ASTD copy I used differs from a fuller one: 95 of that copy's 2,483 binary tweets (3.8%) have no exact match in mine, and 31 of mine have none in theirs. All labels agree on shared tweets and the class balance is the same. I kept mine.

**HARD size mismatch?**
The repo README says 93,700 rows, the file has 105,698. Papers quote 93,700 and 94,052 for the balanced version. I documented the real number and cannot say which version a paper used, so my HARD numbers are not comparable with published ones.

**What about licenses?**
None of the four states one. I treat them as research use, cite the papers, do not track dataset texts in the repository (they remain in earlier git history), and set the model card license to `other`.

**Why store raw text?**
Preprocessing differs by model: the baseline tests normalization on and off, and each transformer has its own tokenizer. Only whitespace is normalized.

**What is your dedupe rule?**
Drop every copy of a text that has conflicting labels, keep one copy of the rest. LABR: 2,338 test rows became 2,283 after removing texts also in train (55). HARD: 105,698 rows became 103,034.

**Why split HARD by hotel?**
Reviews of one hotel share names and room details, and a deployed model sees new hotels. Side effects: validation is 54.6% positive against 49.1% for train, and my scores are not comparable with random-split published numbers.

**Tell me about the HARD leak in 60 seconds.**
A table of each review's first word against its label showed words like "استثنائي" almost always positive and "ضعيف" almost always negative. Crossing the first segment with the star rating showed one phrase per star level, for example "استثنائي" is 5 stars in 6,724 of 6,727 reviews. The nine phrases match about 43% of reviews, and a lookup that gives each phrase its majority label is right about 99% of the time on those rows (my arithmetic from the first crosstab, before one phrase turned out to be a user title). TF-IDF scores 0.941 with it and 0.871 without it on the same rows and splits. Every system loses 5 to 10 points, so I report both variants.

**How sure are you that these are Booking.com's labels?**
It is an inference from the pattern. I did not verify it in the HARD paper.

**What is wrong with your fix?**
The rule compares after stripping quote marks, so it also removes quoted user titles equal to a phrase. That is 6,185 rows (6.0%). Every review opens with either a quoted user title or an unquoted phrase from the list, with no exceptions. A stricter rule (strip only unquoted phrases) changed the TF-IDF score by +0.56 points, so I kept the data and documented the flaw. Encoders and LLMs were not rerun with it.

**I earlier thought the quote pattern was dead. What happened?**
My first test counted only curly quotes and missed straight ones. With both, the pattern is almost perfect.

**Is LABR split by book?**
No, by review: the hub copy has no book id. Reviews of one book can sit in train and test, which may inflate scores. Not measured.

**Why a 20,000-row HARD training subsample?**
Six runs on 81,906 rows were too slow on one laptop GPU, and HARD would dominate the combined model. Scores are probably a little below full-data scores. Not tested.

## Evaluation

**Why macro-F1?**
Class imbalance in ASTD and LABR's full set, and one number across datasets. On the balanced test sets accuracy is about the same.

**How do you decide a gap is real?**
Chapter 6 rules: below half the interval width is noise. On ASTD and AJGT only about 5 points or more counts. Then a paired bootstrap on identical rows where possible. Supported means the interval excludes 0.

**Why 3 seeds on ASTD and AJGT only?**
Their test sets have 242 and 180 rows, so one seed is noise. LABR and HARD have 2,283 and 10,707 rows, and each seed costs time. Training variance there is unmeasured.

**Show a failure of epoch selection.**
AraBERT on ASTD, seed 2: validation peaked at epoch 1 (0.8770) and the test score was 0.8306, while seeds 0 and 1 scored 0.8786 and 0.8904. Validation had 242 tweets, which cannot separate epochs a few points apart. That run makes AraBERT's ASTD standard deviation 0.032.

**The LABR 300-row sample looked odd.**
TF-IDF scores 0.8833 on it against 0.8155 on all test rows, a shift of 6.8 points. A simulation of 10,000 random samples of that shape matched it in 2 of 10,000 draws. I found no defect in how it was drawn, so I call it a rare draw. I do not use the LABR sample for any claim about LLMs against TF-IDF.

**Where did test data influence decisions?**
Honest list: MARBERT was chosen for serving partly because it led on test. The combined model was kept after per-dataset test scores were visible, with a rule written beforehand ("within noise"). The strict-rule check scored TF-IDF on test, for documentation. The AJGT smoke test was scored on test.

**Did you test significance?**
Late, with paired bootstrap on identical rows: all three encoders beat TF-IDF on LABR and both HARD variants. AraBERT and MARBERT beat it on ASTD. On AJGT every interval includes 0. Thirty intervals, no correction. Six had lower bounds under 0.2 points.

## The baseline

**Describe the baseline and what it taught.**
TF-IDF (word 1 to 2 grams or character 2 to 5 grams), normalization on or off, logistic regression with C in {1, 10, 100}: 12 configurations per dataset, chosen on validation. Character n-grams won everywhere. Normalization had no consistent effect. It sets what fine-tuning buys: +3 to +12 points.

**Was the baseline tuned fairly?**
Mostly. I first searched C in {1, 10} and extended it when ASTD's best sat at the edge. For raw HARD the best C is the lowest value tried (1), so a smaller C was never tested. The effect is probably small.

## Encoders

**Why these three models?**
AraBERT: formal Arabic text. MARBERT: tweets, including dialects. XLM-R: a multilingual baseline. The result did not follow the simple story: MARBERT, the tweet model, also led on reviews.

**Which is best?**
MARBERT is best or within 0.3 points of best on all five sets. The paired test supports it over AraBERT on LABR only.

**What was the training recipe?**
AdamW, learning rate 2e-5, weight decay 0.01, 10% warmup then linear decay, batch 32, bf16 autocast, gradient clipping 1.0, 5 epochs on tweets, 4 on LABR, 3 on HARD. Best epoch by validation macro-F1. No hyperparameter search.

**What is the risk of no search?**
Every score could move, especially on the larger datasets, and a fair ranking between close models is not guaranteed.

**Explain the XLM-R story.**
XLM-R trailed on tweets with higher final training loss (0.23 to 0.31 against 0.02 to 0.10), so I guessed it was undertrained. One fixed alternative (lr 3e-5, 10 epochs, batch 16) helped AJGT (0.8685 to 0.9037) and hurt ASTD (0.8367 to 0.8067). Three settings changed at once, so I cannot say which mattered. Both rows are reported. The gap to MARBERT shrinks with training size, a hypothesis I did not test.

**Which hardware limits mattered?**
8 GB of VRAM, about 6.6 GiB usable. XLM-R on reviews peaked at 6.0 GiB with batch 16 and accumulation. XLM-R large needs 8.3 GiB for weights and optimizer alone. A 5070 needs a PyTorch build for CUDA 12.8, and I checked it with a bf16 matrix multiplication.

**What did the tokenizer checks show?**
On all LABR test reviews AraBERT has 2.1% unknown pieces, mostly common words with tanween (جداً, كثيراً). MARBERT lowercases and folds hamza, and has 8.4% unknown pieces on ASTD, mostly `#`, `_` and a straight quote. XLM-R has none. AraBERT's card recommends its own preprocessing, which I did not use, so its scores may be understated.

**What did the 128-token run show?**
MARBERT on LABR at max length 128 scored 0.8844 against 0.8799 at 256. Paired difference +0.44 [-0.48, +1.37]. No detectable effect. MARBERT was pretrained on sequences of 128 tokens, so I wondered about longer positions.

**Why a plain loop and not `Trainer`?**
Every step is visible and explainable, and it avoids API changes between versions. Cost: no resume or per-epoch checkpoints.

**What did MLflow give you?**
Per-run parameters, per-epoch curves and final metrics, in a local SQLite file. It is not in git, so the JSON files in `results/runs` are the durable record.

## LLM evaluation

**How did you evaluate the LLMs?**
Qwen, gpt-oss-120b and ALLaM on Groq. Eight training examples (4 and 4, fixed), a one-word instruction, temperature 0. AJGT and ASTD in full, 300-row samples for LABR and HARD (same rows for both HARD variants). Invalid replies count as errors.

**Results?**
Only Qwen on ASTD matches the best encoder: 0.8966 against MARBERT's 3-seed mean of 0.8902. Paired against each MARBERT seed the difference is -0.2, about +0.5 and +1.7 points, and the seed-0 interval is [-3.5, +3.1], so they are level. gpt-oss is below MARBERT on all five sets. ALLaM, the Arabic specialist, is no better than TF-IDF on ASTD (0.7522 against 0.7708). On LABR and HARD samples no LLM is detectably above TF-IDF.

**Why ALLaM and not Jais?**
Jais at 13B needs about 8 to 9 GB in 4-bit, more than the laptop has, and the Kaggle route needed custom code I could not verify. ALLaM was on the API I already had. Jais was dropped, not tried.

**What happened with the quota?**
I planned from request limits and per-minute token limits. The binding limit was 200,000 tokens a day per model (Qwen and gpt-oss), which I only saw in an error. A `grep` filter hid the progress and the script looked frozen for hours. Fixes: stop on a daily-limit message, run without filters, one queue per model. The test runs took about three days.

**Why were there invalid replies?**
gpt-oss returned empty generations (7 to 10% on the tweet sets), ALLaM sometimes wrote off-task text ("as an AI assistant..."). Invalid replies cost gpt-oss 4.6 points on AJGT and 3.0 on ASTD, so they explain part of its gap, not all.

**What flaw do the HARD prompts have?**
The shots on raw HARD contain the leading phrase, so the model may learn the shortcut from the examples. Not tested.

**How did you choose the prompt language?**
English against Arabic on 60 validation tweets, with a rule fixed beforehand (Arabic only if it wins by 5 points). English won for Qwen and ALLaM. gpt-oss crashed twice on the Arabic run, and English was applied without being tested.

## Dialect and error analysis

**How did you analyze dialects?**
No dataset has per-row dialect labels, so I tagged test rows with a third-party classifier (a MARBERTv2 fine-tune), kept confident ones (0.7 or more) and compared MSA against non-MSA scores.

**What is the result and what does it mean?**
No gap beyond noise for the encoders on LABR and HARD (intervals include 0: HARD within about 1 point, LABR within about 3). ASTD is too small: XLM-R is 9.4 points lower on Egyptian tweets, with interval [-0.6, +19.9]. A null result is weak here, because label noise pushes any gap toward zero.

**How good is the dialect classifier?**
Unvalidated. It labels short Jordanian tweets mostly wrong (only a third of AJGT is tagged Levantine). Agreement with a second classifier (kappa): AJGT 0.71, LABR 0.72, ASTD 0.47, HARD 0.36. Both are MARBERTv2 fine-tunes, so agreement is consistency, not accuracy.

**Why is HARD's Gulf label not a dialect finding?**
28% of HARD rows are tagged Gulf, mostly formal text with Gulf place names and currencies, so the label tracks hotel region, which the hotel split also tracks.

**Describe your error analysis honestly.**
I read 61 MARBERT errors (all 15 on AJGT, all 21 on ASTD, 25 of 274 on LABR). An AI assistant tagged them: 30% no clear sentiment, 21% mixed sentiment, 15% gold label arguably wrong, 21% needs outside context, 13% dialect vocabulary. It is indicative, with one annotator. I have a blind sheet to re-tag.

**What did the error statistics add?**
A quarter to a half of MARBERT's errors are rows all four systems miss. Border ratings are hardest (2 and 4 stars). Errors are less confident than correct answers but 57 to 82% of them come with confidence 0.9 or more. Negative-labeled rows err more often than positive-labeled ones on LABR and HARD.

**What does calibration show?**
Overconfidence: on LABR rows with confidence 0.7 to 0.99 are right 65% to 88% of the time. The API's `confidence` field is a ranking signal, not a probability.

## The combined model

**Why one model?**
A service sees any text, and routing to four specialists needs a classifier that is unreliable on short text. One artifact is simpler.

**What did it cost?**
LABR is 1.6 points lower than the specialist (paired interval [-2.8, -0.4]). ASTD is lower in all three seed comparisons (-3.0, -2.3, -1.1) with seed-0 interval [-6.7, +0.5]. AJGT and HARD no detectable difference.

**How did you build the training set?**
LABR, ASTD, AJGT, and HARD without the descriptor, capped at 20,000 rows (63% of training rows). Original splits kept so each test set equals the benchmark's. Two AJGT tweets also in LABR were dropped as duplicates, with identical labels.

**Why is the pooled score 0.9004 not a good headline?**
79.8% of the test rows are HARD.

End of part 3.
