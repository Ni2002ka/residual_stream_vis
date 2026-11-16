# Experiment B: is the answer readable before the chain of thought?

You are helping run one experiment for a short paper about `residual-stream-vis`,
a browser tool that shows a language model's residual stream as a layer × token
grid and reads every cell with two lenses. This file is your brief. Read the
repo's `README.md` first, especially "What the page shows" and "Chain of thought".

## The claim to test

When an instruction-tuned model answers a two-hop question step by step, the
final answer is often already readable in the residual stream **before the model
writes any reasoning step**. It fails for some kinds of question (orderings and
comparisons, e.g. "who is the shortest?").

What we've seen so far, on `google/gemma-3-4b-it`: for "The Eiffel Tower is in a
country. What is the capital of that country? Think step by step.", both lenses
rank ` Paris` #1 at the newline that closes the question (layer 26 of 34). Same
for the built-in aunt and ball-in-box questions, not for "who is the shortest?".
That's 4 prompts. This experiment turns it into a measured result.

## Ground rules

- Put all work in `experiments/cot_early_answer/`. Don't edit anything under
  `src/residual_stream_vis/`; if the library seems wrong, stop and say so.
- Work on a branch, not `main`. Don't push without asking.
- Ask before downloading anything big. Gemma-3-4B-it is about 8 GB plus its
  J-lens; it needs a GPU (`cuda`, or `mps` on a Mac).
- Keep results reproducible: greedy decoding, fixed seeds, every number in the
  summary comes from a file the script wrote.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # should pass offline in seconds
```

## The library calls you need

```python
import residual_stream_vis as rsv
from residual_stream_vis.edit import word_token
from residual_stream_vis.flow import track

model, tok = rsv.load_model("google/gemma-3-4b-it")             # picks cuda/mps/cpu
lens = rsv.JacobianLens.from_pretrained("google/gemma-3-4b-it")  # published lens

word_token(tok, "Paris")        # (token id, text): the single token the model writes, leading space preferred
rec = rsv.record(model, tok, full_text)          # one forward pass; special tokens kept
rec["tokens"], rec["ids"]                        # per position
t = track(rec, model, tok, " Paris", lens)
t["logit"][layer][pos], t["jlens"][layer][pos]   # rank of the word, 1 = top; jlens is None at layers it doesn't cover
t["n_tokens"]                                    # must be 1, or the question is unusable
```

Layers are residual-stream indices: 0 = embeddings, *i* = output of block *i − 1*.

**Generating** copies `generate()` inside `make_server` in
`src/residual_stream_vis/server.py`. Do it the same way so results match the viewer:

```python
text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
if tok.bos_token and text.startswith(tok.bos_token):
    text = text[len(tok.bos_token):]            # the tokenizer adds it again
enc = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
start = enc["input_ids"].shape[1]               # first response position
full_text = text + tok.decode(out[0, start:], skip_special_tokens=False)
```

Check that `rsv.record(model, tok, full_text)["ids"][:start]` equals the prompt's
ids; if retokenization shifts positions, use the generated ids directly.

## Step 1: the dataset (`questions.jsonl`)

60–100 two-hop questions, about equal across these categories:

| category | example |
| --- | --- |
| `landmark_capital` | The Eiffel Tower is in a country. What is the capital of that country? |
| `family` | Tom's father is Bob. Bob's sister is Alice. Who is Tom's aunt? |
| `containment` | I put a red ball inside a blue box, and the box inside a green bag. What color is the container the ball is directly inside? |
| `ordering` | Anna is taller than Ben. Ben is taller than Carl. Who is the shortest? |
| `person_language` | (person → birthplace/country → language) |

Every question ends with "Think step by step." Each row:
`{"id", "category", "question", "answer", "distractor"}`.

- `answer` and `distractor` must each be **one token** under `word_token`
  (check `track(...)["n_tokens"] == 1`). No numbers: Gemma writes them digit by digit.
- `distractor` is a plausible wrong answer of the same type (another capital,
  another name in the question, another color in the question).
- Vary names, landmarks and colors so no two questions share an answer more than twice.

## Step 2: run (`run.py`)

For each question:

1. Generate as above. Save the response text.
2. Grade: is `answer` the final answer? Do it with a simple rule (last
   occurrence of a candidate word), then hand-check every row the rule
   isn't sure about. Store `correct: bool`.
3. Record `full_text` and track both `answer` and `distractor` under both lenses.
4. Mark positions:
   - `q_end`: the last token of the user's question text
   - `start`: the first response position
   - `written`: the first response position where the answer token appears
     (from the ids; `None` if never written)
5. Save per question: ids, tokens, the three positions, and the two rank grids
   (`answer`, `distractor`) × two lenses. `.pt` or `.npz` is fine; they're big.

## Step 3: measure (`analyze.py`), correct answers only

- **Readable before reasoning**, per lens: does the answer rank #1 at any layer
  at any position in `[q_end, start]` (the end of the question and the chat
  template's tokens before the response)?
- **How early**: the first position `p ≥ q_end` where the answer ranks #1 at
  any layer, as `(p − start) / (written − start)` (≤ 0 means before any
  reasoning); and the layer where that first happens.
- **Control**: the same two numbers for the `distractor`. The claim only holds
  if the answer is readable early much more often than the distractor.
- Also try a looser threshold (rank ≤ 5) and report both.

## Step 4: report (`results/`)

- `per_question.csv`: one row per question with everything above.
- `summary.md`: a table per category with n, % correct, % readable before
  reasoning (logit lens / J-lens / distractor), median relative position,
  median layer. Then the failure cases grouped by what they share, with
  2–3 sentences on the pattern.
- Two figures: a rank heatmap (layers × positions) for one clear success and
  one failure, with `q_end`, `start` and `written` marked. Screenshots from
  the viewer's "Highlight a word" are a fine first version:
  `residual-stream-vis google/gemma-3-4b-it`.

## Stretch goal (only if everything above is done)

For 10 successes, change the first reasoning step to something false (the
viewer's "Edit this transcript", or in code: cut the response after step 1,
replace it, continue generating). Does the final answer change? This tests
whether the written steps are used at all.

## Check-ins

- **Oct 28**: pipeline runs end to end on 5 questions.
- **Nov 4**: full run done, first table and plots.
- **Nov 9**: final numbers and `summary.md`.

When something surprises you (a whole category fails, the lenses disagree,
the distractor is readable as often as the answer), write it down in
`summary.md`. Those notes are as useful as the numbers.
