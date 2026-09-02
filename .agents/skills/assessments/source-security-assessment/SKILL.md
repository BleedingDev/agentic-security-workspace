---
name: source-security-assessment
description: Security-review a source repository, dependency tree, container, or infrastructure configuration.
---

Run `/scope-security-test` unless the scope ledger matches the target identity, assessment ID, and review date; run `/plan-security-assessment` unless its target identity, assessment ID, and review date match that ledger. Map entry points, trust boundaries, sensitive data, authorization gates, and high-risk source-to-sink paths before scanning. Run `/semgrep`, `/trivy`, and `/osv-scanner` where their evidence lanes apply; treat results as candidates requiring reachability and applicability checks. Route service, mobile, and native components to their assessment skills. Run `/verify-security-finding` on every material candidate.

Complete when every planned control and high-risk path has a terminal coverage state, resolved dependencies match the assessed artifact, and material root causes have retest criteria.
