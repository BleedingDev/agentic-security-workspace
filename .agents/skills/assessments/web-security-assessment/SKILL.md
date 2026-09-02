---
name: web-security-assessment
description: Web pentest an authorized browser-facing website or web application.
---

Run `/scope-security-test` unless the scope ledger matches the target identity, assessment ID, and review date; run `/plan-security-assessment` unless its target identity, assessment ID, and review date match that ledger. Map visible and hidden routes, identities, inputs, state transitions, and high-risk business flows before testing. Run `/katana` for crawl evidence, `/burp` for controlled request comparisons, and `/nuclei` for narrow stack-matched candidates. Run `/api-security-assessment` for machine-consumed endpoints and `/verify-security-finding` on every material candidate.

Complete when every planned route, role, input, and business flow has a terminal coverage state and automation limitations are explicit.
