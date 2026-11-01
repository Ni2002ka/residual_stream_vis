# Edit directions: logit lens (unembed) vs J-lens

Strength: push alpha = 1.0 × |cell|, swap strength = 2.0.

- google/gemma-3-4b-it country-capital: 81 one-token items, 81 checked, 79 answered correctly
- google/gemma-3-4b-it antonym: 2219 one-token items, 316 checked, 200 answered correctly
- google/gemma-3-4b-it english-french: 2555 one-token items, 255 checked, 200 answered correctly
- google/gemma-3-4b-it product-company: 33 one-token items, 33 checked, 26 answered correctly
- google/gemma-3-4b-it legs: 31 one-token items, 31 checked, 31 answered correctly

| model | task | edit | n | unembed best (layer) | jlens best (layer) | random best | unembed @ jlens's layer | jlens @ unembed's layer | McNemar p @ jlens's / unembed's layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google/gemma-3-4b-it | antonym | push | 100 | 100% (9) | 100% (27) | 1% | 100% | 7% | 1 / 2e-28 |
| google/gemma-3-4b-it | antonym | swap | 100 | 6% (1) | 0% (1) | 0% | 6% | 0% | 0.031 / 0.031 |
| google/gemma-3-4b-it | country-capital | push | 79 | 100% (9) | 94% (30) | 0% | 100% | 0% | 0.062 / 3.3e-24 |
| google/gemma-3-4b-it | country-capital | swap | 100 | 6% (1) | 73% (20) | 0% | 2% | 0% | 1.5e-19 / 0.031 |
| google/gemma-3-4b-it | english-french | push | 100 | 100% (9) | 97% (30) | 1% | 100% | 1% | 0.25 / 3.2e-30 |
| google/gemma-3-4b-it | english-french | swap | 100 | 7% (1) | 1% (20) | 0% | 0% | 0% | 1 / 0.016 |
| google/gemma-3-4b-it | legs | push | 31 | 100% (10) | 100% (16) | 3% | 100% | 16% | 1 / 3e-08 |
| google/gemma-3-4b-it | legs | swap | 100 | 54% (1) | 17% (21) | 1% | 1% | 2% | 0.00014 / 6.1e-15 |
| google/gemma-3-4b-it | product-company | push | 25 | 100% (7) | 100% (30) | 4% | 100% | 0% | 1 / 6e-08 |
| google/gemma-3-4b-it | product-company | swap | 95 | 22% (2) | 45% (21) | 1% | 16% | 0% | 2.5e-05 / 9.5e-07 |

## Swaps vs activation patching (diagnose.py)

Ceiling: the entity cell copied from a prompt naming the new entity. Handoff: the first layer where copying the cells after the entity beats copying the entity cell. Swap rates here are on diagnose.py's own pairs, at its strength.

| model | task | n | ceiling (layers before handoff) | handoff layer | jlens swap best (layer) | ceiling there | jlens / ceiling | unembed swap best (layer) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google/gemma-3-4b-it | antonym | 20 | 80% | 21 | 0% (1) | 85% | 0% | 0% (1) |
| google/gemma-3-4b-it | country-capital | 20 | 99% | 22 | 75% (20) | 95% | 79% | 20% (2) |
| google/gemma-3-4b-it | english-french | 20 | 88% | 21 | 0% (1) | 95% | 0% | 25% (2) |
| google/gemma-3-4b-it | legs | 20 | 91% | 21 | 10% (8) | 100% | 10% | 65% (1) |
| google/gemma-3-4b-it | product-company | 20 | 95% | 22 | 50% (20) | 80% | 62% | 15% (1) |
