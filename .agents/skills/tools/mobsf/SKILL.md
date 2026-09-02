---
name: mobsf
description: MobSF-analyze an Android package through the persistent local API service.
---

Hash the package, run `mobsf-agent scan INPUT OUTPUT_JSON`, and triage the report into candidates that name their manifest, configuration, or code evidence. Pass every material candidate to `/verify-security-finding`.

Complete when the report is valid JSON and every material candidate points to an independently inspectable location.
