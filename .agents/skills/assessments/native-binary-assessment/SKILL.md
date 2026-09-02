---
name: native-binary-assessment
description: Reverse engineer or security-assess an EXE, DLL, ELF, Mach-O, firmware image, or native library.
---

Run `/scope-security-test` unless the scope ledger matches the target identity, assessment ID, and review date; run `/plan-security-assessment` unless its target identity, assessment ID, and review date match that ledger. Identify the artifact by provenance, cryptographic hash, format, architecture, and protections. Keep facts, hypotheses, experiments, observations, and open questions distinct.

Run `/capa` for capability leads and `/ghidra` for function, cross-reference, and call-path evidence. Treat capability matches as hypotheses. Record packed, obfuscated, encrypted, or unextractable content as `untested` with reason and impact. Execute unknown or suspicious code only in an authorized isolated environment, with the experiment tied to one unresolved hypothesis. Correlate each runtime observation to that hypothesis and a static function or call path; otherwise mark it inconclusive. Run `/verify-security-finding` on every material candidate.

Complete when every planned entry point, trust boundary, and risky behavior is evidenced, refuted, or recorded as an explicit gap.
