# Word swaps: lens directions vs directions from other prompts (google/gemma-3-4b-it)

From `word_swaps.py`, 20 pairs per task, edit at the entity token. Success at the best layer, then at layer 12: cosine between the edit's change and patching's change, and the change's size relative to the cell. jlens = J-lens swap (s = 2); mean_task = mean cell difference over 4 other prompts of the task; mean_generic = mean difference over 4 generic sentences ("He said big"); patch = copy the cell from the prompt with the new word. For legs, every prompt shares its shots, so mean_task equals patch.

```
  antonym          jlens 0% (L1) cos 0.08 size 0.02   mean_task 95% (L3) cos 0.85 size 0.12   mean_generic 95% (L4) cos 0.50 size 0.16   patch 85% (L1) cos 1.00 size 0.14
  english-french   jlens 0% (L1) cos 0.08 size 0.01   mean_task 95% (L1) cos 0.87 size 0.12   mean_generic 95% (L8) cos 0.39 size 0.12   patch 95% (L1) cos 1.00 size 0.11
  country-capital  jlens 75% (L20) cos 0.26 size 0.04   mean_task 100% (L1) cos 0.86 size 0.09   mean_generic 100% (L1) cos 0.73 size 0.11   patch 100% (L1) cos 1.00 size 0.10
  legs             jlens 20% (L12) cos 0.23 size 0.03   mean_task 100% (L1) cos 1.00 size 0.08   mean_generic 100% (L1) cos 0.42 size 0.11   patch 100% (L1) cos 1.00 size 0.08
```

## Size controls (`--kinds`, same 20 pairs)

jlens_sized / unembed_sized: the lens swap's change rescaled to mean_generic's size; random_sized: a random direction of that size; generic_small: mean_generic's change shrunk to the J-lens swap's size. Leg count has 5 possible answers, so chance there is about 20%.

```
  antonym          jlens 0% (L1) cos 0.08 size 0.02   mean_generic 95% (L4) cos 0.50 size 0.16   jlens_sized 5% (L9) cos 0.08 size 0.16   unembed_sized 5% (L1) cos 0.02 size 0.16   random_sized 0% (L1) cos -0.00 size 0.16   generic_small 0% (L1) cos 0.50 size 0.02
  english-french   jlens 0% (L1) cos 0.08 size 0.01   mean_generic 95% (L8) cos 0.39 size 0.12   jlens_sized 0% (L1) cos 0.08 size 0.12   unembed_sized 20% (L3) cos 0.02 size 0.12   random_sized 0% (L1) cos 0.00 size 0.12   generic_small 5% (L8) cos 0.39 size 0.01
  country-capital  jlens 75% (L20) cos 0.26 size 0.04   mean_generic 100% (L1) cos 0.73 size 0.11   jlens_sized 70% (L11) cos 0.26 size 0.11   unembed_sized 30% (L3) cos 0.05 size 0.11   random_sized 0% (L1) cos -0.00 size 0.11   generic_small 30% (L13) cos 0.73 size 0.04
  legs             jlens 20% (L12) cos 0.23 size 0.03   mean_generic 100% (L1) cos 0.42 size 0.11   jlens_sized 50% (L14) cos 0.23 size 0.11   unembed_sized 30% (L4) cos 0.04 size 0.11   random_sized 20% (L3) cos -0.00 size 0.11   generic_small 25% (L11) cos 0.42 size 0.03
```
