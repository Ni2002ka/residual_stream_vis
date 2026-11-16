# Brief: polish Scenario A of the paper

You're helping polish one section of a short paper. Work only on **Section 4,
"Scenario A: Two Kinds of Edit Direction"** in `paper/main.tex` (from
`\section{Scenario A` up to `\section{Scenario B`), including its table
(`tab:directions`) and the Fig. 2 caption (`fig:directions`). Don't touch other
sections, the figures themselves, or anything under `experiments/` or `src/`.

## The paper

A PacificVis 2027 VisNotes short paper (6 pages including references, IEEE VGTC
template, due Jan 4, 2027). It presents **residual-stream-vis**, an interactive
viewer that shows a language model's residual stream as a grid (one column per
token, one row per layer). Each cell can be read with two lenses (the logit lens
and the J-lens), traced, and edited, with the layers above rerun. The paper is a
tool paper: three usage scenarios backed by numbers, no user study.

Macros: `\LL` = "logit lens", `\JL` = "J-lens", `\todo{}` = red draft marker,
`\cref{}` for references.

## What Scenario A says

To make a model answer *Tokyo* instead of *Paris* for "Q: France / A:", a user can
edit in two ways:

- **push**: add a word's direction to the last token's cell (change the output directly);
- **swap**: replace *France* with *Japan* at the France token (change what the
  model reasons from, and let it work out *Tokyo*).

Each edit needs a direction for a word. The **logit-lens direction** (the word's
row of the output matrix) suits the push; the **J-lens direction** (that row
times the layer's average Jacobian J) suits the swap, but only for fact lookups.

## The numbers (all Gemma-3-4B-it). Do not change any number without a source.

Sources: `experiments/edit_directions/results/summary.md` (main tables, both
runs) and `experiments/edit_directions/results/direction_cosine.json`.

Sample sizes (correctly answered items with one-token words, capped at 100):

| task | push items | swap pairs |
| --- | --- | --- |
| country–capital | 79 | 100 |
| product–company | 25 | 95 |
| antonym | 100 | 100 |
| English–French | 100 | 100 |
| leg count | 31 | 100 |

The patching ceiling and the swap/patching cosines come from a **separate run of
20 pairs per task** (`diagnose.py`). Say so whenever they're compared with the
main numbers.

- **Push**, first layer reaching ≥90% success: logit lens 6–9 on every task,
  J-lens 16–30. At the logit lens's best layer the J-lens push gets 0–16%,
  p < 1e-7 (exact McNemar) on every task. Random pushes: ≤4% at any layer.
- **Cosine between the two directions** for the same word (208 words): 0.07 at
  layer 8, 0.31 at 20, 0.51 at 27, 0.79 at 33. The J-lens push catching up
  late reflects the two directions converging.
- **Answer at the entity token** (J-lens top 5, layer 20; best layer in parentheses), from
  `answer_at_entity.md`: country–capital 58% (77%), product–company 19% (58%), antonym 8%
  (15%), English–French 3% (5%), leg count 0%; another item's answer: 0% everywhere. The
  entity word itself at layer 12: 98% / 96% for countries / products, 27% / 22% for antonym /
  translation words, 77% for animals.
- **Swap**, at each direction's best layer (strength s = 2):

| task | logit lens | J-lens | random | ceiling |
| --- | --- | --- | --- | --- |
| country–capital | 6% (L1) | 73% (L20) | 0% | 99% |
| product–company | 22% (L2) | 45% (L21) | 1% | 95% |
| antonym | 6% (L1) | 0% | 0% | 80% |
| English–French | 7% (L1) | 1% (L20) | 0% | 88% |
| leg count | 54% (L1) | 17% (L21) | 1% | 91% |

  - Logit-lens swap at the J-lens's best layer: 2% (country–capital) and 16%
    (product–company), p = 1.5e-19 and 2.5e-5.
  - "79% and 62% of the ceiling" is the J-lens swap divided by patching on the
    **20-pair run** (75% vs 95%, 50% vs 80%). It is **not** 73% ÷ 99%.
  - The swap removes 1–5% of the cell's length. Its change lines up with
    patching's change at cosine 0.16–0.27 (J-lens, fact tasks), 0.06–0.13
    (J-lens, antonym and translation) and about 0.05 (logit lens).
  - Failed swaps keep the original answer rather than producing a wrong one.
  - Logit-lens swaps at layers 1–3 are close to replacing the input word: Gemma
    ties its input and output embeddings.
- **Handoff**: every swap stops working after layer 21–22. From there, copying
  the cells after the entity beats copying the entity cell.
- **Held-out check**: picking the best layer on a random half of the pairs and
  scoring the other half (20 splits) changes rates by at most 5 points.

## Agreed rewrites not yet in the file

The author approved these in conversation. Start from them rather than the
versions currently in `main.tex`.

**"Why the two differ"** (replace the current paragraph):

```latex
\textbf{Why the two differ.} For the same word, the two directions are almost unrelated through the middle of the
model: averaged over the 208 words pushed toward, their cosine is 0.07 at layer~8 and 0.31 at layer~20, and they
become similar only near the top (0.51 at layer~27, 0.79 at layer~33; \cref{fig:directions}c). They stand for
different things. The \LL{} direction is the word as the top layer reads it at this token: added to the last token,
it rides the residual connections to the output almost unchanged, and the model says the word. The \JL{} direction is
what the layers above usually \emph{turn into} the word, which makes it an input to reason with rather than an
answer. At the last token there is nothing left to reason toward, so a \JL{} push fails until the top layers, where
the two directions coincide and it becomes, in effect, a \LL{} push. The converse also holds: the last token is the
wrong place to change what the model reasons with. In two-hop questions, swapping the intermediate state at the last
token fails even with the \JL{} direction, while the same swap at the city token succeeds (\cref{sec:twohop}).
```

**"Swap: use the J-lens direction, for facts"** (replace the current paragraph):

```latex
\textbf{Swap: use the \JL{} direction, for facts.} On the two fact-lookup tasks, the \JL{} swap gives the new
entity's answer for 73\% and 45\% of pairs at its best layer (20 and 21). At those layers the \LL{} swap succeeds on
2\% and 16\% of the same pairs ($p=1.5\times10^{-19}$ and $2.5\times10^{-5}$, exact McNemar), and a random edit of
the same size on at most 1\%; the \LL{} direction does best near the input (6\% and 22\% at layers 1--2), where
swapping it is close to replacing the input word. As a ceiling, we copy the whole entity cell from a prompt that
names the new entity, the most any edit of that one cell can do: on a separate set of 20 pairs, the \JL{} swap reaches
79\% and 62\% of what this patching achieves at the same layer. The swap falls short because it is a small, coarse
edit. It removes only the old word's component, 1--5\% of the cell's length, and its change lines up only loosely
with the change patching makes (cosine 0.16--0.27 for the \JL{} direction, about 0.05 for the \LL{}). When it fails,
the model keeps its original answer rather than producing a wrong one.
```

**Table caption** (replace the `tab:directions` caption):

```latex
\caption{Edits on Gemma-3-4B-it. Push: the first layer from which the pushed word is the top prediction for at
    least 90\% of items. Swap: success at each direction's best layer, and for a random edit of the same size at its
    best layer; ceiling: copying the whole entity cell from a prompt naming the new entity, averaged over the layers
    before the handoff (a separate set of 20 pairs per task). Items pushed per task: country--capital 79,
    product--company 25, antonym 100, English--French 100, leg count 31; swap pairs: 100 per task (95 for
    product--company). Random pushes succeed on at most 4\% of items at any layer. Selecting the best layer is not
    what makes the rates high: choosing it on a random half of the pairs and scoring the held-out half gives rates
    within 5 points of those shown.}
```

## Leave alone for now

- **"Where neither works"**: its explanation is now backed by a test
  (`experiments/edit_directions/results/answer_at_entity.md`). Start from this
  agreed version; polish wording, but keep the numbers and the hedge ("appears
  to"), since the test is correlational:

  ```latex
  \textbf{Where neither works.} On antonym and translation, both directions fail (at most 7\%), although copying the
  entity cell from a prompt with the new word succeeds 80--88\% of the time: the information is in the cell, but one
  word direction does not capture it. Reading the entity token shows why. For the fact lookups, the \JL{} already
  reads the \emph{answer} there: \emph{Paris} is among its top five words at the \texttt{France} token for 58\% of
  country--capital items at layer~20, and the company for 19\% of products. For antonym and translation the answer
  almost never appears at the entity token (8\% and 3\%), nor for leg count (0\%), and a control answer from another
  item never does. So for fact lookups the model resolves the answer on the entity token and later tokens read it
  from there, which is what a swap of the entity can redirect. For the other tasks the operation appears to run at
  the answer position, which reads the word itself from the entity token, and one direction does not stand in for a
  word: on its own token the \JL{} reads \emph{hot} or \emph{dog} for only 22--27\% of items at layer~12, against
  96--98\% for countries and products. On leg count, only the \LL{} direction works, and only at layers 1--3 (54\%),
  where the cell is still close to the token's input embedding (Gemma ties its input and output embeddings), so the
  swap amounts to changing the input word.
  ```

- **Swap strength**: Scenario A uses s = 2, Scenario B uses s = 4. The author
  chose to keep it this way for now. Don't change either.
- **The `\todo{make these the viewer's defaults.}`** at the end of the section
  stays until the code changes.

## What to polish

- Clarity first. Each paragraph should open with its claim in plain words, then
  the numbers. Define every term before using it (push, swap, ceiling, handoff,
  best layer).
- Cut repetition between the Setup paragraph and the opening example.
- Keep it tight: the whole paper must fit in 6 pages including references, and
  it is at 4 now with one section (Scenario C) and one figure (Fig. 1) still to
  come. Aim for this section not to grow.
- Keep the table's numbers and structure. Wording in headers and captions can change.
- Don't add claims the numbers above don't support. If a sentence needs a number
  that isn't listed here, add a `\todo{}` instead of guessing.

## How to work

- The author runs `latexmk -pdf -pvc main.tex` in `paper/`, which rebuilds on
  save. **Don't run latexmk or pdflatex yourself while it's running**; two
  builds collide. Check `paper/main.log` after saving for errors, undefined
  references or overfull boxes: `grep -c "^!\|undefined\|Overfull" paper/main.log`
  should print 0.
- The author pastes LaTeX in themselves, and may prefer you hand back code
  blocks rather than editing the file. Ask which they want.
- Don't commit or push. The branch is `two-hop`; the author has unrelated
  uncommitted work in `src/` and `tests/`. Leave it alone.
- Tone: plain, specific, no hype. Prefer "the J-lens swap succeeds on 73% of
  pairs" over "dramatically outperforms".
