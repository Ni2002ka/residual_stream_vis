# Project Overview

So far, we've built **residual-stream-vis**, a tool that lets you watch a prompt move
through a language model. It shows the model's residual stream (the vector
each token carries from layer to layer) as a grid: one column per token, one
row per layer. Every cell is labeled with the word a _lens_ reads from it,
meaning the word the model would say if it had to answer from that vector
right now. You can click any cell to see where its information came from, and
edit it to see what changes.

There are two lenses:

- **Logit lens**: decode the vector as if it were already at the top of the model.
- **J-lens**: first estimate what the layers above will do to the vector, then
  decode. It often picks up an idea a few layers earlier.

## The paper

We're writing a short paper (a "VisNotes" paper, 6 pages, for IEEE PacificVis
2027, due January 4). It's a tool paper: it shows what the tool does and what
you can find with it. Its evidence is two short studies, each showing
something the tool makes easy to see:

1. **Two kinds of edit direction** (I'm running this one). Pushing a cell with
   the logit lens's direction for a word changes what _that token_ outputs.
   Pushing it with the J-lens's direction changes what _later tokens read_
   from it. Each works for its own job and fails at the other's.
2. **The answer comes before the reasoning** (yours). When a model answers a
   question "step by step", the final answer is often already sitting in the
   residual stream before it writes any of the steps.

## Your experiment

Here's the example that started it. We asked Gemma-3-4B-it: _"The Eiffel Tower
is in a country. What is the capital of that country? Think step by step."_
It writes out "the Eiffel Tower is in France… the capital of France is
Paris." But both lenses already read **Paris** as the top word at the end of
the question, around layer 26 of 34, before the model has written the
word "France". The same happens for a couple of other built-in questions, but
not for "Anna is taller than Ben, Ben is taller than Carl, who is the shortest?",
where the answer only shows up just before it's written.

I just tested four examples. Your job is to find out whether it holds up:

- Build a set of 60–100 two-hop questions across a few categories (landmarks
  → capitals, family relations, objects inside containers, orderings, and one
  more).
- Have the model answer each one step by step, then use the library to check,
  cell by cell, when the answer first becomes readable.
- Compare against a plausible _wrong_ answer, so we know the lens isn't just
  ranking every likely-sounding name highly.
- Work out which kinds of question show the effect and which don't, and why.

A clean result either way is useful. If the effect is strong for some
categories and absent for others, that's a better paper than "it always works".

## What a good result looks like

- A table per category: how often the model is right, and how often/early the
  answer is readable before any reasoning (with each lens, and for the wrong
  answer as a control).
- Two heatmaps: one question where it works and one where it doesn't.
- A page of notes on the failures and anything that surprised you.

## Proposed workflow

Give the technical brief in `CLAUDE.md` to your Claude, next
to this file. Please skim it yourself too, and push back on anything that seems off. You'll know the data better than the brief does.

Spend some time in the viewer itself (`residual-stream-vis
google/gemma-3-4b-it`, then "Let the model answer" and "Highlight a word") before
writing code. It's the fastest way to get a feel for what you're measuring, and your impressions of the tool will be useful in cleaning up the UX for the final submission.

## Timeline

| date   | check-in                                    |
| ------ | ------------------------------------------- |
| Oct 28 | the pipeline runs end to end on 5 questions |
| Nov 4  | full run done, first table and plots        |
| Nov 9  | final numbers and notes                     |

After Nov 9 we'll look at both experiments together and decide what the paper says.
