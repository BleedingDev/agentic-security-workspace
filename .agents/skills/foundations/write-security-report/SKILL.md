---
name: write-security-report
description: Write the evidence-backed report for an authorized security assessment.
---

After no plan row remains `planned`, write `security-artifacts/report.md` once. Include scope and exclusions, methods and versioned profiles, the coverage matrix, assumptions, limitations, root-cause-grouped findings, evidence links, confidence and contextual risk, remediation, retest criteria, residual gaps, and cleanup or data-disposition status. Explain the reason and impact of every `not-applicable`, `untested`, or inconclusive result. Keep candidate disposition (`confirmed`, `refuted`, `inconclusive`) separate from plan execution and test result.

Complete when every claim links to raw evidence, every in-scope surface has a terminal execution state and, when tested, a result, and credentials are absent from the report.
