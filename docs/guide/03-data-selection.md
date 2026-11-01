## Resolved: our ASTD copy differs from a fuller copy

Measured here, after whitespace normalization, comparing `arbml/ASTD` (used) with `Qanadil/ASTD_Arabic_Sentiment_Tweets_Dataset`:

| | arbml (used) | Qanadil |
|---|---|---|
| Negative | 1,642 | 1,684 |
| Neutral | 805 | 832 |
| Positive | 777 | 799 |
| Subjective total | 3,224 | 3,315 |

Qanadil's 3,315 equals the subjective-tweet count reported for the original ASTD in a paper.

- 3,186 texts match exactly, and the labels agree on every one of them.
- 129 Qanadil tweets have no exact match in our copy (61 negative, 34 neutral, 34 positive). For the binary task that is 95 of Qanadil's 2,483 tweets (3.8%).
- 38 of our subjective tweets have no exact match in Qanadil (19 negative, 12 positive, 7 neutral), so neither copy is a subset of the other. Whether the unmatched tweets are different tweets or edited forms of the same tweet was not checked.
- Positive share: 32.1% in ours, 32.2% in Qanadil.

Decision: keep our copy. The class balance is the same and the labels agree, so about 4% of tweets differing should be inside the noise of a 242-tweet test set. That is reasoning, not a measurement. Rebuilding ASTD from the other copy would mean rerunning every ASTD experiment, including the LLM runs, which are limited by daily token caps. The difference is stated in the README limitations.

Interview angle: the check also shows that the labels are consistent across two independently uploaded copies, which is mild evidence that the annotation itself was not corrupted in either upload.
