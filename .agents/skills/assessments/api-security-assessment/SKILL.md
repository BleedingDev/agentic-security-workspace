---
name: api-security-assessment
description: API pentest an authorized REST, GraphQL, RPC, mobile-backend, or other machine-consumed service interface.
---

Run `/scope-security-test` unless the scope ledger matches the target identity, assessment ID, and review date; run `/plan-security-assessment` unless its target identity, assessment ID, and review date match that ledger. Inventory hosts, versions, operations, methods, schemas, authentication, and undocumented or deprecated surfaces.

Build a subject × object × action × field matrix. Use `/burp` for controlled request comparisons, `/katana` where crawling applies, and `/nuclei` only for narrow stack-matched candidates. Test authentication, object/property/function authorization, bounded resource use, sensitive business-flow abuse, outbound requests, and third-party API trust. Run `/verify-security-finding` on every material candidate.

Complete when every planned operation, identity relationship, sensitive business flow, versioned or deprecated surface, and outbound or third-party trust hypothesis has a terminal execution state and, when tested, a result; resource tests stayed within the scope budget.
