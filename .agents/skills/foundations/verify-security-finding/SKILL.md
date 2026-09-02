---
name: verify-security-finding
description: Verify a scanner, static-analysis, or reverse-engineering security candidate before reporting it.
---

A candidate is material when it could change risk, coverage, or remediation; pass every material candidate through this skill. Restate it as attacker-controlled input, missing or bypassed control, reachable operation, and impact. Reproduce it against a benign control or corroborate it through an independent evidence lane. Seek the strongest contrary evidence.

Record `confirmed`, `refuted`, or `inconclusive`, plus asset or artifact identity, prerequisites, minimum safe proof, raw-evidence paths, limitations, confidence, root-cause group, and retest condition.

For a confirmed root cause, search within its planned hypothesis for variants by generalizing one element at a time. Stay inside the plan's request, time, count, and cost bounds; update `plan.md` before testing a new surface or larger budget. Record failed queries and false-positive reasons; send every retained candidate through this skill independently.

Complete when the disposition is supported by evidence and every missing proof is explicit.
