Cause	AJGT (15)	ASTD (21)	LABR (25)
Label noise or no clear sentiment (neutral, factual, news, hashtags, one name)	6	9	3
Mixed or hedged sentiment (praise plus criticism, “I love it but...”)	0	2	11
Gold label arguably wrong	2	3	4
Needs world or political context (stance, sarcasm, religion, politics)	4	6	3
Dialect vocabulary plausibly responsible	3	1	4

Rows can fit two categories, so I put each in its most likely one, and the columns sum approximately.

Examples behind the counts

No clear sentiment: AJGT "صراع متصاعد للهيمنه على سوق الالمنيوم العالميه" (gold negative, a neutral news line), ASTD "نصر محروس" (a bare name, gold negative), “#الفرسان_كورة | اخبار | #محمد_صلاح يقود...” (sports news).
Mixed sentiment (LABR): "الرواية حلوة وعدم الترتيب..." (gold positive, but a complaint follows), and "رواية ممتازة فى اسلوبها ولكنها تدعو إلى الفجور" (gold negative despite "ممتازة").
Stance and context (ASTD): several political tweets ("الانقلاب يريد قمع 50 مليون مصرى...") carry positive gold labels for the author’s cause, and the model reads the grievance wording as negative. That is stance, not tone.
Gold label questionable: ASTD "شرييييفة! #فيفي_عبده" labelled negative, and AJGT "اربد فيها جامعات اكثر من عمان..." labelled positive in a way a reader can accept either way.
Dialect vocabulary: AJGT "صار بدها بطاريه التقوى شحن" (Levantine "بدها" plus slang) and "عيش زي ما بدك ما ترد على حدا ابدا", where a Levantine idiom reads as negative to the model.

In a manual read of 61 MARBERT errors, the most common causes were rows with no clear sentiment, mixed sentiment, stance-dependent labels and questionable gold labels. Dialect vocabulary appeared plausible in about 13% of errors. This is a single-annotator reading of a small sample.
