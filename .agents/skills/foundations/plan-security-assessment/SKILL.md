---
name: plan-security-assessment
description: Plan coverage for an authorized security assessment after scope is known and before active testing.
---

Read `security-artifacts/scope.md`; record its matching target identity, assessment ID, and review date in the plan. Orient before hunting: map assets, entry points, identities and roles, attacker-controlled inputs, trust boundaries, sensitive flows, persistent state, security controls, dependencies, and unresolved assumptions.

Select the applicable versioned standard or testing profile. Write `security-artifacts/plan.md` as a matrix with asset, branch, versioned control/test ID or custom hypothesis ID, applicability and rationale, bounded method, expected evidence, execution state, and result. Execution is `planned`, `tested`, `not-applicable`, or `untested`; a tested row resolves to `pass`, `fail`, or `inconclusive`. Record the reason and impact of every `not-applicable`, `untested`, or `inconclusive` row.

When fuzzing is selected, name the harness target, expected-error oracle, deterministic seed or corpus strategy, CPU/memory/time/network bounds, and crash-provenance output.

Complete when every in-scope surface has an owning assessment skill, a safe evidence plan, and an explicit execution state. Reporting may start only when no row remains `planned`.
