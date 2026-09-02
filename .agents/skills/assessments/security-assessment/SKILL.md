---
name: security-assessment
description: Pentest an authorized target across multiple security surfaces, or coordinate a complete assessment.
---

Run **scope-security-test** unless the scope ledger matches the target identity, assessment ID, and review date. Run **plan-security-assessment** unless its identity and review date match that ledger.

For each **planned** row, run the matching surface skill: **web-security-assessment**, **api-security-assessment**, **android-security-assessment**, **native-binary-assessment**, or **source-security-assessment**. Preserve raw evidence before changing its state. Pass every material candidate through **verify-security-finding**, then set the row to a terminal execution state and result.

On tool failure, preserve the error and diagnose once. Retry only after the cause, inputs, or bounded method changes. Otherwise mark the row **untested** with reason and impact. A scope stop returns to **scope-security-test**; an interrupted run resumes only rows still **planned**.

Run **write-security-report** once after every row is terminal. Complete when the coverage matrix accounts for every in-scope surface, cross-branch findings are grouped by root cause, and cleanup or retained evidence is recorded.
