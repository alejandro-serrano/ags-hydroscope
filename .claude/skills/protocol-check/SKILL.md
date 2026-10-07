---
name: protocol-check
description: Run the independent experiment-auditor subagent over configs, results, splits and the paper, and summarize what must be fixed before a checkpoint (G3 results frozen, G4 full draft, G5 submission).
disable-model-invocation: true
context: fork
agent: experiment-auditor
---

Audit the repository as described in your instructions. Checkpoint being prepared: $ARGUMENTS
(default: G3). Return only the findings table and a one-line verdict: ready or not ready.
