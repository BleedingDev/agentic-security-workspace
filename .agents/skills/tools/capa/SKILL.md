---
name: capa
description: capa-triage a native binary for capability leads before deeper reversing.
---

Save machine-readable results and treat rule matches as leads for `/ghidra`. Route ARM64 ELF samples directly to Ghidra when the local vivisect backend cannot load them. Pass every material capability candidate to `/verify-security-finding`.

Complete when every material capability has a verification disposition and unsupported analysis is recorded as a gap.
