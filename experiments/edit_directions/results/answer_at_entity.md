# Is the answer readable at the entity token? (google/gemma-3-4b-it)

From `answer_at_entity.py`. Fraction of correctly answered items whose word is in the lens's top 5 at the entity token ("Q: France"): best layer, then layers 12 / 20. answer = this item's answer (" Paris"); entity = the entity word itself (" France"); ctrl_answer = another item's answer.

```
  country-capital  n= 60 jlens  answer 77% (L26) 47%/58%   entity 100% (L7) 98%/100%   ctrl_answer 0% (L1) 0%/0%
  country-capital  n= 60 logit  answer 33% (L28) 0%/0%   entity 100% (L27) 0%/3%   ctrl_answer 0% (L1) 0%/0%
  antonym          n= 60 jlens  answer 15% (L25) 3%/8%   entity 62% (L28) 27%/58%   ctrl_answer 0% (L1) 0%/0%
  antonym          n= 60 logit  answer 7% (L25) 0%/0%   entity 42% (L27) 0%/0%   ctrl_answer 2% (L19) 0%/2%
  english-french   n= 60 jlens  answer 5% (L18) 2%/3%   entity 70% (L24) 22%/52%   ctrl_answer 0% (L1) 0%/0%
  english-french   n= 60 logit  answer 2% (L21) 0%/0%   entity 67% (L27) 0%/15%   ctrl_answer 0% (L1) 0%/0%
  product-company  n= 26 jlens  answer 58% (L22) 15%/19%   entity 100% (L7) 96%/100%   ctrl_answer 0% (L1) 0%/0%
  product-company  n= 26 logit  answer 50% (L25) 0%/0%   entity 77% (L24) 0%/0%   ctrl_answer 0% (L1) 0%/0%
  legs             n= 31 jlens  answer 0% (L1) 0%/0%   entity 94% (L25) 77%/84%   ctrl_answer 0% (L1) 0%/0%
  legs             n= 31 logit  answer 0% (L1) 0%/0%   entity 52% (L27) 0%/0%   ctrl_answer 0% (L1) 0%/0%
```

## Does J-lens readability predict J-lens swap success per pair? No.

Joined the J-lens swaps (s = 2, main run) with this file's ranks by (task, word, layer). "Readable" = in the J-lens top 5 at the word's own token. Exact two-sided Fisher test. Coverage: all pairs for country–capital (65/100 with both words), product–company and legs; only 15 and 6 pairs for antonym and English–French.

- Both words readable vs not: country–capital L12 17% (n=59) vs 67% (n=6), p=0.017 (the wrong way); L16 and L20 every pair is readable, yet success climbs 37% → 71%. Legs L20 20% (n=69) vs 3% (n=31), p=0.033; other layers n.s. Antonym: 0/5 readable pairs succeed at L20.
- New item's answer readable at its entity: country–capital L16 17% (n=52) vs 72% (n=25), p=5e-6 (the wrong way); L20 70% vs 70%. Product–company L16 42% vs 14%, p=0.029; L20–21 n.s.

So the task-level link (fact lookups: words and answers readable, swaps work; word tasks: neither) does not hold pair by pair. Within a task, what the J-lens reads at the entity token does not tell you which swaps will work, and the swap's success depends mostly on the layer.
