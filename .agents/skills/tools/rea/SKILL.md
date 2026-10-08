---
name: rea
description: Use REA for native, Android, JavaScript/Electron, .NET, archive, Apple resource, or retained network evidence in this security workspace.
---

# REA

Use REA when the question depends on a shipped artifact, decompilation, application graph, version comparison, or recorded behavior. For source-repository architecture, use repository tools. Read the installed release's `../../reverse-engineer-anything/SKILL.md` and only its relevant reference. If that workflow or the MCP connection is missing, use [the REA setup lane](../../foundations/setup-security-workspace/references/rea.md); keep repairs local to this checkout.

For security testing, run `/scope-security-test` and `/plan-security-assessment` unless the matching ledgers already cover the technique. Identify the target by provenance, hash, format, and architecture. Start with a focused static query; call only advertised tools using their actual schemas. Save complete raw Evidence, IDs, locations, and limitations under the target's `security-artifacts/evidence/`. Separate observations, inferences, and unknowns. Runtime capture follows the declared execution scope and isolation rules.

REA complements the existing lanes. Reuse the installed Ghidra and JADX engines. REA's temporary headless Ghidra projects are separate from the existing GUI bridge. Keep `/ghidra` for attached projects, `/apktool` for resources and smali, `/mobsf` and `/capa` for independent checks, `/burp` for HTTP testing, and `/frida` or `/adb` for instrumentation and device control. REA and a direct call to the same engine are not independent corroboration.

Pass material security candidates through `/verify-security-finding`. Close opened native sessions with `close_binary`; record any cleanup failure and ownership before finishing.

Complete when the requested hypothesis has evidence and explicit limitations, required companion-tool evidence is correlated, and owned sessions are closed or have a documented cleanup boundary.
