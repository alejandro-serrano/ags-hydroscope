---
name: sprint-status
description: Agile status report - reads open and closed GitHub issues of the current milestone with gh, compares them with the plan checkpoints (G1 Wed Oct 7, G2 Fri Oct 9, G3 Wed Oct 14, G4 Fri Oct 16, G5 Wed Oct 21) and proposes what to cut if a checkpoint is at risk.
disable-model-invocation: true
allowed-tools: Bash(gh issue list *) Bash(gh issue view *) Bash(git log *)
---

## Issues

!`gh issue list --state all --limit 100 --json number,title,state,labels,milestone,assignees`

## Recent commits

!`git log --since="7 days ago" --oneline`

## Instructions

Today is the date of the latest commit or the current date if you know it. Produce, in Spanish:
1. Progress per line (Modelos, Datos, Artículo) and per checkpoint G1–G5: done / at risk / late.
2. Blockers: issues In progress for more than 2 days or with unmet dependencies.
3. If a checkpoint is at risk, propose cuts in this order: Could → Should (SSL variant, state map) →
   fewer epochs for ALL models. Never propose cutting the generalization analysis.
4. Three concrete tasks for tomorrow.
