---
name: qmd-style
description: Write experiment reports as Quarto (.qmd) notebooks with plotly plots in the MatX technical writing style. Use this skill whenever the user asks to write a report, write up results, create an analysis, document an experiment, or produce any .qmd file. Also trigger when the user says "write this up", "make a report", "document these results", or references the qmd/quarto format for a new report.
---

# QMD Report Style

Write Quarto (.qmd) experiment reports that read like short technical memos: compressed, declarative, data-driven. The model is the MatX research blog — every sentence either defines something, derives something, or reports a measurement.

Before writing, fetch and read https://matx.com/research/leaky_quantization to internalize the target prose style. This is the reference document — match its voice, density, and structure.

## Writing voice

The voice is impersonal and declarative. State things as facts with the reasoning immediately adjacent to the claim — no preamble, no hedging, no buildup. The reference style is the MatX research blog (e.g. "Future Leakage in Block-Quantized Attention"). Study these examples from that post.

### Flat assertions, zero hedging

Every claim is either a definition, a logical consequence of the previous sentence, or backed by a measurement in the same document.

**From MatX — definitions stated as one-sentence identities:**
- "Future leakage is when information from positions >i may influence the logits at position i."
- "In causal language modeling, the final logits at position i must depend only on tokens at positions <=i."

No "Let us first define..." or "To understand this, we need to..." — just state the definition.

**From MatX — claims stated as flat facts:**
- "However, training with block quantized attention can break causal modeling."
- "Quantizing P is safe despite blocking along token positions, since the causal mask zeros out future probabilities."
- "Leakage occurs when query position i and value position j share a quantization block position."

No "we believe", "it appears", "arguably", "it seems likely". If the claim is true, state it. If it isn't backed by data, cut the sentence.

**From MatX — counterintuitive results, stated flatly:**
- "Despite the Leaky model's parallel loss being lower than the Fixed model's, the autoregressive loss was worse."
- "The Leaky model's gap in loss between parallel and autoregressive modes indicates that it was reliant on future signal."

No "Surprisingly", no "Interestingly", no exclamation. State the paradox, then explain it mechanistically.

### "We" is for actions, not opinions

**From MatX:**
- "We present a fix that enables training with MXFP4 in both attention and the attention gradient."
- "To validate this fix, we trained a pair of 1B-parameter models on the C4 dataset."

"We" is always attached to a concrete verb: present, trained, aim. Never "we think", "we believe", "we feel", "we suspect".

### Logical connectives, not narrative ones

**From MatX:**
- "With this procedure, s and consequently $q_i$ depend on all the pre-quantized elements."
- "So if the quantization block spans across different token positions, quantization enables the later tokens in the block to influence earlier tokens."
- "So we can prevent future leakage by using unquantized P and V when multiplying the i-th tile of P with the i-th block of V."

The connective tissue is logical — "so", "since", "consequently", "because". Not narrative — "next", "then we noticed", "interestingly", "as it turns out".

### Paragraphs

Short. 2-4 sentences typical. Some paragraphs are a single sentence. If a paragraph exceeds 4 sentences, split it.

### Qualifiers to delete

None of these words appear in a well-written report: "arguably", "perhaps", "it seems", "it is likely that", "it appears", "we suspect", "in our opinion", "to some extent", "relatively", "quite", "fairly", "somewhat", "surprisingly", "interestingly". Delete on sight.

### Section structure

Every section does exactly one thing. The MatX post has 6 sections arranged as a logical chain: define the invariant (causality) -> explain the mechanism (block quantization) -> show how the mechanism breaks the invariant -> identify exactly when -> fix it -> prove the fix works. Each section exists to set up the next one.

Transitions between sections are implicit. The heading starts with a fresh declarative sentence. No "Having established X, we now turn to Y" or "Now let us consider..."

### Data presentation

Tables carry the quantitative story. Prose interprets a table in one sentence — naming the pattern, not restating numbers.

**From MatX — a compact results table does the talking:**

| Model | Parallel | Autoregressive | Gap |
|-------|----------|----------------|-----|
| Leaky | 2.56 | 2.66 | +0.10 |
| Fixed | 2.64 | 2.64 | 0.00 |

The prose then says: "The Leaky model's gap in loss between parallel and autoregressive modes indicates that it was reliant on future signal." One sentence. Not a paragraph restating every cell.

Use inline math ($x_i \approx q_i \cdot s$, $b_i = \lfloor i/k \rfloor$) where it's clearer than prose. Math is embedded in the prose flow, not set apart in numbered equation blocks. No theorem/lemma/proof structure.

### What to cut

- Rhetorical questions in prose (a section heading can be a question — MatX uses "When does quantized V cause future leakage?")
- "It is important to note", "Note that", "Importantly"
- Bullet lists for argumentation (bullets only for enumerating items like the three cases $b_i = b_j$, $b_i > b_j$, $b_i < b_j$)
- Summaries mid-document ("To summarize what we've seen so far...")
- Related work sections, lengthy motivation, apologies about limitations
- Discovery narrative ("we noticed", "we were surprised to find", "during our experiments we observed")

## Quarto format

### Frontmatter

```yaml
---
title: "Descriptive title — what happened, not what was attempted"
subtitle: "one-line elaboration if needed"
date: YYYY-MM-DD
categories: [relevant, tags]
---
```

### Code blocks

All plotly code blocks use `#| echo: false` — the reader sees the chart, not the code. Code-fold in the quarto config makes the source available if they want it.

```
\```{python}
#| echo: false
import plotly.graph_objects as go
# ...
fig.show()
\```
```

When showing config, logs, or terminal output, use fenced YAML/text blocks with Quarto callout annotations (`# <1>`, `# <2>`) for inline commentary.

### Plotly theme

Read `references/plotly-theme.md` for the project's standard dark theme colors and layout boilerplate. Every chart in a report uses this theme for visual consistency with the Quarto dark mode.

Key rules for charts:
- Use `make_subplots` for related metrics that share an x-axis
- Color-code metrics consistently across charts via a `colors` dict
- `mode="lines+markers"` for time series
- Legend at bottom center, horizontal orientation
- Subplot titles name the metric, axis titles name the unit
- No decorative elements — every visual element encodes data

### Tables

Use markdown tables, not plotly tables. Keep them compact — no column wider than needed.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `kl_coeff` | 0.0 | No anchor to pretrained distribution |

### Report structure

A report is a logical chain. Common shape:

1. **Result summary** — 2-3 sentences. What happened, measured.
2. **Setup** — Config, model, environment. Tables for parameters.
3. **Results** — Charts and tables. Prose names the pattern, doesn't restate numbers.
4. **Analysis** — Why the results look the way they do. Each cause gets its own paragraph or sub-section.
5. **Next steps** — Table of what changes. No speculation about outcomes.

This structure flexes — a post about a single bug might be: problem, mechanism, fix, validation. The invariant is: each section does one thing, the sections form a logical chain, every claim is backed by data or derivation.

## Registering the report

After writing a new .qmd file in `docs/`, update `docs/_quarto.yml` to add the report to the navbar menu under "Reports".
