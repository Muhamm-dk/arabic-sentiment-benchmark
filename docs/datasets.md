# Datasets

Task: binary sentiment (negative vs positive) on all four datasets, so results are comparable.

## LABR
- Source: Hugging Face `mohamedadaly/labr` (official balanced split). `elsheikhams/labr-ar` (full 63,257 reviews) used for cross-checking.
- Size: 11,760 train, 2,935 test (official). Full set: 63,257.
- Labels: 1 to 5 stars (HF label index = stars minus 1). Binary: 1-2 negative, 4-5 positive, 3 dropped.
- Variety: book reviews, mixed MSA and dialect.
- License: not yet verified, check the dataset card.
- Notes: official split is exactly balanced (2,352 per class train, 587 per class test). Full set is skewed toward 5 stars (23,778 vs 2,939 for 1 star). 65 texts appear in both train and test, removed from test in the loader. 300 duplicate rows in the official data.

## HARD
- Source: GitHub `elnagara/HARD-Arabic-Dataset`, file `data/balanced-reviews.zip`. Paper: Elnagar, Khalifa, Einea (2018), Springer.
- Size: 105,698 rows in the balanced file. The repo README says 93,700, which is out of date; `data/readme.txt` says to use the zip as the latest. The full unbalanced set (373,750 reviews) is not used.
- Labels: ratings 1, 2, 4, 5 only (no 3 stars). Binary: 1-2 negative (52,849), 4-5 positive (52,849).
- Variety: hotel reviews from Booking.com, MSA and dialect.
- Format: tab-separated, UTF-16 little-endian with BOM and CRLF. Read with `encoding='utf-16', quoting=3`. Columns: no, Hotel name, rating, user type, room type, nights, review.
- License: not yet verified, check the repo.
- Notes: 1,662 duplicate review texts, handled in the loader. No official split, so we create a seeded stratified split.

## ASTD
- Source: Hugging Face `arbml/ASTD`.
- Size: 9,694 tweets, single train split (no official split).
- Labels: Objective 6,470, Negative 1,642, Neutral 805, Positive 777. Binary keeps Negative and Positive only: 2,419 tweets, 68% negative.
- Variety: tweets, mostly Egyptian dialect with some MSA.
- License: not yet verified, check the dataset card.

## AJGT
- Source: Hugging Face `komari6/ajgt_twitter_ar`.
- Size: 1,800 tweets, single train split (no official split), 900 positive and 900 negative.
- Labels: Negative, Positive.
- Variety: tweets, Jordanian dialect with some MSA.
- License: not yet verified, check the dataset card.
