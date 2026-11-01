# Two-hop recall: google/gemma-3-4b-it

Readable = in the lens's top 5.

- us: 46 one-token items, 25 answered correctly. Wrong answers: Houston → Houston, Chicago → Chicago, Seattle → Seattle, Spokane → Spokane, Tacoma → Washington, Miami → Florida, Orlando → Orlando, Tampa → Florida, Jacksonville → Jacksonville, Philadelphia → Philadelphia, Cincinnati → Cincinnati, Baltimore → Maryland, Memphis → Tennessee, Knoxville → Knoxville, Savannah → Georgia, Fairbanks → Alaska, Oakland → California, Shreveport → Louisiana, Norfolk → Norfolk, Provo → Utah, Ogden → Utah
- world: 51 one-token items, 30 answered correctly. Wrong answers: Toronto → Toronto, Vancouver → Victoria, Sydney → Sydney, Melbourne → Melbourne, Perth → Perth, Barcelona → Barcelona, Frankfurt → Frankfurt, Marseille → Marseille, Zurich → Switzerland, Geneva → Switzerland, Casablanca → Morocco, Krakow → Krakow, Porto → Portugal, Manchester → Manchester, Liverpool → Liverpool, Karachi → Karachi, Antwerp → Antwerp, Medellín → Medellín, Córdoba → Córdoba, Novosibirsk → Novosibirsk, Haifa → Tel

## When each word becomes readable (per item, first layer in the top k)

| template | token | lens | n | state ever | capital ever | ctrl state ever | state before capital | state same layer | capital before state | median first layer: state / capital |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| us | city | logit | 25 | 100% | 0% | 0% | 0% | 0% | 0% | 22 / nan |
| us | city | jlens | 25 | 100% | 32% | 0% | 20% | 0% | 12% | 7 / 23 |
| us | last | logit | 25 | 100% | 100% | 12% | 92% | 8% | 0% | 25 / 28 |
| us | last | jlens | 25 | 100% | 100% | 32% | 92% | 0% | 8% | 23 / 27 |
| world | city | logit | 30 | 97% | 0% | 0% | 0% | 0% | 0% | 24 / nan |
| world | city | jlens | 30 | 100% | 47% | 0% | 17% | 10% | 20% | 15 / 9 |
| world | last | logit | 30 | 100% | 100% | 0% | 10% | 57% | 33% | 25 / 25 |
| world | last | jlens | 30 | 100% | 100% | 10% | 23% | 20% | 57% | 25 / 25 |

## Edits (success = the new capital is top-1)

| template | edit | direction | strength | n | best (layer) |
| --- | --- | --- | --- | --- | --- |
| us | city@city | jlens | 2.0 | 40 | 57% (17) |
| us | city@city | jlens | 4.0 | 40 | 65% (20) |
| us | city@city | unembed | 2.0 | 40 | 8% (1) |
| us | city@city | unembed | 4.0 | 40 | 8% (19) |
| us | patch_city | patch | - | 40 | 100% (1) |
| us | patch_last | patch | - | 40 | 100% (33) |
| us | random@last | random | 2.0 | 40 | 0% (1) |
| us | random@last | random | 4.0 | 40 | 0% (1) |
| us | state@city | jlens | 2.0 | 40 | 90% (21) |
| us | state@city | jlens | 4.0 | 40 | 92% (21) |
| us | state@city | unembed | 2.0 | 40 | 25% (21) |
| us | state@city | unembed | 4.0 | 40 | 38% (22) |
| us | state@last | jlens | 2.0 | 40 | 0% (1) |
| us | state@last | jlens | 4.0 | 40 | 8% (20) |
| us | state@last | unembed | 2.0 | 40 | 8% (21) |
| us | state@last | unembed | 4.0 | 40 | 2% (1) |
| world | city@city | jlens | 2.0 | 40 | 70% (20) |
| world | city@city | jlens | 4.0 | 40 | 85% (20) |
| world | city@city | unembed | 2.0 | 40 | 8% (1) |
| world | city@city | unembed | 4.0 | 40 | 12% (3) |
| world | patch_city | patch | - | 40 | 98% (4) |
| world | patch_last | patch | - | 40 | 100% (25) |
| world | random@last | random | 2.0 | 40 | 0% (1) |
| world | random@last | random | 4.0 | 40 | 0% (1) |
| world | state@city | jlens | 2.0 | 40 | 88% (20) |
| world | state@city | jlens | 4.0 | 40 | 98% (18) |
| world | state@city | unembed | 2.0 | 40 | 5% (14) |
| world | state@city | unembed | 4.0 | 40 | 25% (21) |
| world | state@last | jlens | 2.0 | 40 | 15% (22) |
| world | state@last | jlens | 4.0 | 40 | 22% (21) |
| world | state@last | unembed | 2.0 | 40 | 8% (25) |
| world | state@last | unembed | 4.0 | 40 | 5% (15) |

Counterfactual prompt answered with the new capital: us 100%, world 100%

