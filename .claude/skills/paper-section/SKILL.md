---
name: paper-section
description: Draft or revise one section of the IEEE paper with the paper-writer subagent. Usage - /paper-section <introduction|related|data|methodology|results|discussion|conclusions>
disable-model-invocation: true
context: fork
agent: paper-writer
---

Write or revise the section: $ARGUMENTS.

- File: `paper/sections/<section>.tex`, included from `paper/main.tex`.
- For `results`, first make sure `paper/tables/` and `paper/figures/` are up to date (they come from
  the /results-tables skill) and only reference them.
- Target length: introduction 0.75 page, related 0.75, data 0.5, methodology 1, results 1.5,
  discussion 1, conclusions 0.25 (two-column IEEE pages).
- End with a list of `\todo{}` items you could not resolve from the sources.
