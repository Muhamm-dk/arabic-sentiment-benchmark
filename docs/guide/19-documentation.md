# Chapter 19: Documentation and honest reporting

## What

The writing that surrounds the code: the README, the model card, the dataset notes, the error analysis, this guide, and the habits that kept them honest. This chapter reviews them as a reader would and records what was found.

## Why it matters

- **A portfolio repository is read in a few minutes.** The README decides whether anyone opens a second file. It must say what was done, what was found and how sure the author is, on the first screen.
- **Claims that survive checking are the point.** This project made more than a dozen claims that later measurements corrected. A reader who finds one unmarked correction trusts everything less, and a reader who sees them reported openly trusts the numbers more.
- **Two audiences.** The README and model card are for strangers. The guide and the interview notes are for the author, and the notes stay out of the public repository.

## Inventory

| File | Reader | Contains |
|---|---|---|
| `README.md` | strangers, recruiters | findings, tables, how to reproduce, how to serve, deployment, limitations, citations |
| `docs/model_card.md` (also on the Hub) | model users | data, training, results with intervals, limits |
| `docs/datasets.md` | people reproducing the work | sources, sizes, label mapping, licenses, processing decisions, the HARD leak, the ASTD copy |
| `docs/error_analysis.md` | reviewers | 61 hand-read errors, categories, limits of the annotator |
| `docs/guide/` | the author | chapters 0 to 19 and appendices |
| `results/*.json, *.csv, *.md` | anyone | every score file, paired results, the tables |
| `docs/interview/` | the author only | local, not tracked (see below) |

Tracked in git: code, tests, Dockerfile, run scripts, `pyproject.toml`, `uv.lock`, score files, docs. Not tracked: `data/`, `outputs/` (weights), prediction files, LLM replies, dialect tags, logs, `mlflow.db`, the guide's build output. The reasons: dataset texts have no stated license (Chapter 3), weights live on the Hub, and logs hold account metadata.

## The README, reviewed as a reader

What works:
- The first screen names the task, the methods and the model link, then three findings with numbers.
- Every results table says whether it is full-test or a 300-row sample, and says why they must not be compared.
- The limitations list is specific and sits near the end, where a skeptical reader looks for it.
- Seeds, subsamples and the AI-tagged error analysis are stated as such.

Problems found on review, and what was done:

| Problem | Fix |
|---|---|
| Finding 2 said "5 to 10 points". The 10 was TF-IDF on a 300-row sample, while 7.0 is its full-test figure | Reworded: "about 5 to 8 points (TF-IDF 7.0 on the full test set; the LLMs are measured on 300 rows)" |
| No finding about the LLMs, half the project | Added finding 4 |
| The Reproduce block missed the HARD download, the combined model and the tests | Rewritten, with a layout note and a pointer to the run scripts |
| The deployment section had no example call | Added a `curl` example with the key header |
| Citations carried "verify details before citing" | Verified (below) and written in full |
| The AJGT row said 1,438 training rows. That is the combined model's figure | Corrected to 1,440 |
| The error analysis said "the model author's assistant", which hid that the annotator was an AI | Named |
| The cold-start figure was one measurement | Two measurements and the scale-down time |

Still weak: the README is long (about 120 lines), and a reader sees the leak finding only after the key findings. A short summary table at the top (best model per dataset with its score) would help. No figures or plots exist. The README has no badge or status line, and no contact line.

## Licenses and citations

- **Code:** MIT (a `LICENSE` file from the repository's creation).
- **Datasets:** none states a license. The LABR hub card's licensing section reads "[More Information Needed]", the ASTD card has no license field, the AJGT card lists "unknown", and the HARD readme only asks for citation. Treated as research use.
- **Model:** the card says `license: other` and explains why. A permissive license on weights trained on data with unclear terms would overclaim.
- **Citations, verified against the ACL Anthology, the Springer record and the dataset readme:**
  - LABR: Aly and Atiya, ACL 2013 (short papers), pp. 494-498. An arXiv version (1411.6718) lists a third author, Nabil, on a differently titled version. I did not compare the two, so the README cites the ACL paper.
  - HARD: Elnagar, Khalifa and Einea, 2018, Springer, from the dataset readme.
  - ASTD: Nabil, Aly and Atiya, EMNLP 2015, pp. 2515-2519.
  - AJGT: Alomari, ElSherif and Shaalan, IEA/AIE 2017, Springer, pp. 602-610.
- **Note:** the ASTD paper names its fourth class "subjective mixed", and the hub copy calls it "Neutral". Whether they are the same class was not checked. Both are dropped in the binary task.
- **Model citations:** MARBERT (verified from the UBC repository), AraBERT and XLM-R by arXiv id.

## Honest-reporting practices, and what they cost

1. **Write the prediction before the run, then score it.** Used for tables, tests and checks throughout. Some failed (the XLM-R tuning, the length trend in the errors, the gpt-oss valid-only score), and those are in the chapters as failures. Cost: a longer chapter.
2. **Fix the rule before the result.** The prompt language, the tuned XLM-R configuration, the data-keep rule for the HARD fix.
3. **Report both rows.** Both XLM-R configurations, both HARD variants.
4. **State what a number does not cover:** one seed, a 300-row sample, test-row sampling only.
5. **Separate measured from remembered.** "I measured" against "from memory".
6. **Correct in public.** A wrong claim is corrected in the chapter, the README and the notes in one pass, with the right number.
7. **A score counts only when its summary file exists,** after a partial run printed an F1 of 1.0 on one row.
8. **Keep secrets out of files and chats.** Hidden prompts for tokens, unset afterwards. One key was pasted into a chat by mistake, rotated, and the tracked files and results were scanned.

## Claims corrected during the project

| Claim | What measurement showed | Chapter |
|---|---|---|
| HF Docker Spaces are free | Not selectable on the free account | 18 |
| HARD has about 490,000 reviews | The balanced file has 105,698. The README says 93,700. Papers vary | 3 |
| The curly-quote idea is dead | The test missed straight quotes. Every HARD review opens with a quoted title or an unquoted phrase | 5 |
| `hard_notitle` is a conservative lower bound | The rule strips 6.0% of rows (quoted titles), so it is not a bound | 4, 5 |
| Tuning XLM-R will fix it | It helped AJGT and hurt ASTD | 10 |
| C was interior on every dataset; word features lose 1 to 3 points | Raw HARD's best C is the lowest tried. The gap is 0.2 to 6.2 points | 7 |
| Encoders beat TF-IDF everywhere; MARBERT best on 4 of 5 | XLM-R ties on AJGT. MARBERT has the top score on 3 | 10, README |
| The encoders are not separable on HARD | Paired tests: slightly, by under 1 point | 10 |
| About 10% of LABR is truncated at 256 | About 5% for AraBERT and MARBERT | 3, 8 |
| ALLaM's refusals are on negative tweets | Both polarities | 11 |
| gpt-oss would score about 0.92 on valid replies | 0.876 | 11 |
| Quota limits were requests per day | Tokens per day (200,000) | 11 |
| 1,438 is the post-dedupe AJGT size | Two tweets also in LABR were dropped by the combined build | 15 |
| XLM-R's fixed training memory is 4.4 GiB | 4.14 GiB | 8 |
| Error rates are similar across dialects | True on ASTD only | 14 |
| Positive labels hide criticism more than negative ones | Retracted: negative-labelled rows err more | 14 |
| The `hard_notitle` dialect rows were fine | A text-join bug dropped 43% of them | 13 |
| One cold-start measurement is enough | Two so far, 27.5 s and 26.0 s | 18 |
| The error annotator was "the model author's assistant" | An AI assistant (Claude) | README |

## Repository hygiene checks

Run these before sharing the repository. The key scans use patterns for the three kinds of token handled in this project (Groq, Hugging Face, GitHub). They match the prefix and at least 20 more characters.

```bash
git grep -n -E "(gsk_|hf_|ghp_)[A-Za-z0-9]{20,}" $(git rev-list --all) | head
grep -rIl -F "$(cat ~/.sentiment_api_key)" . --exclude-dir=.git --exclude-dir=.venv | head
git log -S"$(cat ~/.sentiment_api_key)" --oneline | head
git ls-files | grep -E "\.(parquet|jsonl|log|db|safetensors|pt)$"
git count-objects -vH
```

Each of the first three must print nothing. The fourth lists tracked data-like files (expect none). The last shows the repository size. Not yet checked here: the output of any of them.

Other items:
- **History contains early commits with dataset texts** (prediction and reply files, removed from tracking later) and run logs with an organization id. Not credentials. Rewriting history was judged not worth a force-push.
- **Commit style:** conventional prefixes (`feat`, `docs`, `fix`, `chore`) and one change per commit, mostly. The count is well above the planned 12 (`git log --oneline | wc -l`).
- **`ruff` was installed and never run.** `uv run ruff check src tests` would show style issues (for example import order in `app.py`). Not run.
- **No CI.** Tests run only locally and skip without weights.

## Two-minute reader test

Open the repository on GitHub, signed out. Check that tables render, the links to the Hub model and the study guide work, the demo URL answers `/health`, and the first screen says what the project found. Then read only the limitations list and ask whether any claim in the findings is contradicted there. A mismatch means a fix.

## Not done

- A summary table at the top of the README, and plots.
- A second pair of eyes on the writing.
- Human re-tagging of the error sheet, which would turn an AI-only analysis into a checked one.
- CI, linting, pinned dependencies.
- A short blog-style write-up of the leak finding.

## Check yourself

1. Name three problems the README review found, and how each was fixed.
2. Why does `license: other` fit the model card?
3. Which citation did I have to check against a version I had not compared, and what did I do?
4. Why are predictions written down before runs, and what happened when some failed?
5. Why does the notes folder stay out of the public repository, and what does stay public?
6. Which secrets checks would you run before sharing, and what must they print?
7. Pick two corrected claims from the table and explain how each was caught.
