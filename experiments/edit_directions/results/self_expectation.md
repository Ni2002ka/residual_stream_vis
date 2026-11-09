# Does a word that predicts itself get read by the J-lens? (google/gemma-3-4b-it)

From `self_expectation.py`: 125 words (25 per task). Self-expectation = mean log p(word) over a fixed neutral passage after a frame holding the word, minus the same with the other words in the frame (nats). J-lens rank of the word at its own token in "He said {w}". Spearman with n = 25 per task: |rho| > 0.40 is roughly p < 0.05.

```
Per task: median self-expectation (nats) and median J-lens rank of the word at its own token
  country-capital  expect  8.98   L1      3  L4      1  L8      0  L12      0
  product-company  expect  8.11   L1      0  L4      0  L8      0  L12      0
  legs             expect  8.17   L1      4  L4      3  L8      0  L12      2
  antonym          expect  5.13   L1    217  L4     96  L8      8  L12     35
  english-french   expect  6.59   L1    438  L4    121  L8      7  L12     15

Spearman(self-expectation, -J-lens rank): pooled, then within each task
  L 1  pooled  0.59  country-capital -0.14  product-company  0.03  legs  0.44  antonym  0.01  english-french  0.42
  L 4  pooled  0.61  country-capital -0.20  product-company -0.03  legs  0.55  antonym -0.03  english-french  0.29
  L 8  pooled  0.57  country-capital  0.02  product-company -0.06  legs  0.30  antonym -0.17  english-french  0.55
  L12  pooled  0.53  country-capital -0.18  product-company -0.06  legs  0.38  antonym -0.14  english-french  0.52
```

Across tasks the two line up (pooled rho 0.53-0.61): names raise their own later probability by 8-9 nats and are read at rank 0-4; antonym and translation words by 5-7 nats and are read at ranks 7-438. Within tasks: positive for legs (0.30-0.55) and English-French (0.29-0.55), null for antonym (-0.17 to 0.01); countries and products have no spread in rank to correlate. Partial support; the pooled correlation may also carry confounds of word type (e.g. token frequency), which this does not control.

## Within antonyms: rarity, not self-expectation

The antonym words the J-lens cannot read at layer 4 are common words with many senses (final, flat, spread, spending, spare, heat: ranks 3,500-28,000); the readable ones are rarer and specific (spiral, impractical, decorative, collateral, unpopular, prose: ranks 2-42). With the token id as a rough rarity proxy (BPE merge order; a higher id is usually a rarer piece), Spearman(id, -J-lens rank) at L4 / L12: antonym 0.38 / 0.48, English-French 0.36 / 0.43, legs 0.48 / 0.65, country-capital -0.05 / 0.24, product-company 0.03 / 0.23, pooled 0.52 / 0.62. Rarer words are read better within every non-entity task, including antonym, where self-expectation showed nothing. Names are the exception: France (id 7001) is as common a piece as spread (7028) and is read at rank 0. The id is only a proxy for frequency.

## Number of senses (WordNet 3.0 synsets, via nltk)

Spearman(number of senses, J-lens rank of the word at its own token), so positive = more senses, read worse; then the same controlling for the token id (rarity proxy). 16 product names have no WordNet entry and are left out.

```
task              n    median senses | L4    L12  | controlling token id: L4   L12
antonym           25   4             | 0.45  0.46 | 0.37  0.37
english-french    25   6             | 0.71  0.38 | 0.71  0.35
legs              25   4             | 0.24  0.32 | -0.06 -0.09
country-capital   25   1             | -0.07 0.30 | -0.11 0.32
product-company   9    2             | 0.05  0.40 | -0.06 0.25
pooled            109  3             | 0.54  0.52 | 0.43  0.38
```

Antonym words by J-lens rank at L4 (readable first), with sense counts: spiral 10, wonderful 1, hawk 6, impractical 2, decorative 1, collateral 5, unpopular 1, dual 3, prose 2, exceptional 3, inverse 3, void 8, constructive 2, dotted 6, disperse 6, coarse 3, servicing 4, confidential 4, heat 11, spare 13, final 5, flat 24, spending 5, spread 23, exclusion 4.

Words with more senses are read worse within antonym and English-French, also with rarity held fixed (0.35-0.71). Not within legs once rarity is controlled. Countries have one sense each, so there is no spread to test.
