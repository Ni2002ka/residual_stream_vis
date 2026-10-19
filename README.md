# residual-stream-vis

Type a prompt and watch it move through a language model. A local page shows the
model's residual stream as a grid, one column per token and one row per layer,
and lets you read, trace and edit any cell.

```bash
residual-stream-vis gpt2                    # or google/gemma-3-1b-pt
```

The page opens at http://127.0.0.1:8000 and runs on your machine only.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/residual-stream-vis gpt2
```

The first run downloads the model and its J-lens from the Hugging Face Hub.
After that, `HF_HUB_OFFLINE=1` skips the network checks.

## What the page shows

**The grid.** Columns are tokens, rows are layers: the input embedding at the
bottom, the model's output at the top. Each cell is the residual stream vector
for that token after that layer, labeled with the word a lens reads from it:
*if the model had to say the next word from this vector, which word would it
pick?* Switch between two lenses:

- **Logit lens:** decode the cell with the model's own final norm and
  unembedding, as if it were already at the top.
- **J-lens:** first multiply by `J`, the average Jacobian from that layer to the
  top (measured once on web text), then decode. It estimates what the layers
  above would do to the cell, so it often reads an idea earlier.

At the top layer both lenses show the model's real prediction. At the bottom
there is no prediction yet, so the readouts are junk; the page explains why.

**Click a cell** to see both lenses' top words, how the block changed it
(kept from below, attention write, MLP write), which earlier tokens and heads
the attention write came from, and three views of the flow:

- **Into this cell:** what one block pulled in. Curved lines are attention
  from earlier tokens, weighted by how much each source actually passes on;
  the straight line is what the token kept from itself.
- **Trace back / forward:** chain those shares block by block to follow the
  cell's information back to the input, or one cell's information forward to
  the output. An approximation (a norm-weighted attention rollout); it tends
  to pile onto the first tokens, and "Ignore the first token" drops that
  attention sink.

**PCA.** Below the grid, every token at one layer, projected onto that layer's
top 2 or 3 principal components (L2-normalized vectors). It follows the
selected layer; click a point to select its cell. Axes are refit at every
layer, so compare tokens within a layer.

**Edit a cell and rerun.** Change the selected cell, let every layer above it
recompute, and compare before and after:

- **Push toward a word:** add the word's direction, scaled by the cell's size.
- **Swap one word for another:** remove one word's direction and add another's.
- **Copy from another prompt:** replace the cell with a donor prompt's cell
  at the same layer (activation patching).
- **Add a difference:** add the difference between two prompts' cells (a
  steering vector).

Word edits can use either the **logit-lens direction** (the word's unembedding
row) or the **J-lens direction** (that row pulled back through `J`). They're
good at different things:

| edit | logit-lens direction | J-lens direction |
| --- | --- | --- |
| push the last token of "The Eiffel Tower is located in the city of" toward ` London`, Gemma-3-1B, layer 8 | **London 100%** | 0% |
| swap the animal at its own token in "How many legs does a … have?", Gemma-3-1B, 24 pairs, layer 8 | 0 / 24 | **5 / 24** answer with the new animal's count |

To make a token *output* a word, use the logit-lens direction: the residual
connections carry it straight to the top. To change a concept that *later
tokens read*, use the J-lens direction: `J` adds up a cell's effect at its own
position and every later one. Try the built-in leg-count example on Gemma:
swap spider → ant at the ` spider` token, layer 8, and the answer goes from
8 to 6 with J-lens directions but stays 8 with logit-lens ones.

**What changed** shades every cell by how much it moved: only the edited cell
and cells above and to the right of it can change.

**Chain of thought.** With an instruction-tuned model, **Generate a response**
lets the model answer step by step (using its chat template); the grid then
covers your message and its response. **Track a word** colors every cell by
how highly a lens ranks one word, so you can see whether the answer is already
there before the model writes its reasoning:

```bash
residual-stream-vis google/gemma-3-4b-it      # ~8 GB download; needs a GPU
```

On Gemma-3-4B-it, for "The Eiffel Tower is in a country. What is the capital of
that country? Think step by step.", both lenses rank ` Paris` #1 at the newline
that closes the question (layer 26 of 34), before any of the written steps. The
same holds for the built-in aunt and ball-in-box questions, but not for "who is
the shortest?". Track one-token words: Gemma writes numbers digit by digit.

## Models

Blocks are found automatically for GPT-2, Llama, Mistral, Qwen, Gemma and most
decoder-only models. The J-lens loads automatically for models with a lens
published by Neuronpedia (`rsv.PRETRAINED_LENSES`: GPT-2 and Gemma 3); others
get the logit lens only, or fit their own:

```python
lens = rsv.fit_lens(model, tok, texts)   # ~100+ texts of ~128 tokens
lens.save("my_lens.pt")
```

```bash
residual-stream-vis my/model --lens my_lens.pt
```

A prompt is one forward pass plus a lens readout of every cell. On an Apple
GPU (picked automatically) Gemma-3-1B takes about 3 seconds; on CPU the readout
against its 262k-word vocabulary takes minutes, so use a GPU for Gemma.

Command-line options: `--port`, `--host`, `--device`, `--lens auto|none|PATH`,
`--no-browser`, and extra arguments become one-click example prompts.

## From Python

```python
import residual_stream_vis as rsv

rsv.serve("gpt2")                                        # the same page

model, tok = rsv.load_model("gpt2")
lens = rsv.JacobianLens.from_pretrained("gpt2")
data = rsv.trace(model, tok, "The Eiffel Tower is located in the city of", lens)
rsv.flow_html(data, "flow.html")                         # a static page, no server needed

prompt = "The Eiffel Tower is located in the city of"
d = rsv.word_direction(model, tok, " London", layer=8, kind="unembed")
r = rsv.steer(model, tok, prompt, d * rsv.cell(model, tok, prompt, 8).norm(), layer=8)
r.prob(" London")                                        # (before, after)
```

`rsv.patch`, `rsv.swap` and `rsv.steering_vector` cover the other edits;
`rsv.Edit` with the `rsv.editing(model, edits)` context manager applies edits
to any forward pass. Layers are residual stream indices (0 = embeddings, *i* =
output of block *i − 1*). Positions skip special tokens such as `<bos>` unless
`skip_special_tokens=False`.

## How it works

One forward pass with hooks records each block's input and output, what
attention wrote (`attention + MLP = output − input`, checked against the
MLP's own output in the tests), every head's attention weights, and the size of
what each token passes on through each head. Attention shares are attention
weight × that size, as in Kobayashi et al.'s norm-based analysis. The J-lens
estimator follows Anthropic's [reference implementation](https://github.com/anthropics/jacobian-lens),
so fitted and published lenses are interchangeable. The method is from
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, 2026).

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The tests build tiny random GPT-2, Llama and Gemma-3 models, so they run
offline in seconds.
