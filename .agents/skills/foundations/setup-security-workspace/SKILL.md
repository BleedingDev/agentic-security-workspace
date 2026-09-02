---
name: setup-security-workspace
description: "Install and configure the full local security toolchain for this checkout when the user invokes setup-security-workspace."
---

# Setup Security Workspace

Run once per checkout before the first assessment. Invocation authorizes routine tool, package, runtime, and local-service installation needed by this skill. Install the full profile by default; use a smaller profile only when the user explicitly asks. Record progress in **security-artifacts/setup.md** so setup is resumable and every installed component is removable later.

## 1. Inspect

Read **README.md** and **AGENTS.md**. Identify the OS, architecture, agent host, available package managers and privileges, then enumerate every tool under **.agents/skills/tools/** plus its required runtime, data, service, and MCP integration. Preserve compatible existing installations rather than upgrading them merely because a newer release exists.

Classify each component as **ready**, **missing**, **manual**, or **failed**. For every component, record: component, ownership (**pre-existing**, **setup-owned**, or **manual**), official source, install method and command, discovered version, paths or service identity, status, evidence, and exact uninstall or cleanup command. Complete when every full-profile component is classified and has an official source.

## 2. Install

For each missing component, read [official tool sources](references/tool-sources.md), then consult its current official installation instructions. Prefer the operating system's supported package manager; otherwise use the vendor's supported installer, release, or runtime manager. Install automatically, in dependency order, and verify each executable or service immediately. Write its ownership and removal instructions to the ledger before continuing.

Do not pause for routine installation confirmation. Use already-available privilege safely. When elevation, license acceptance, GUI interaction, driver approval, or device connection cannot be automated, complete every safe precursor and request one exact user action.

On failure, preserve the command, output, and exit status, then diagnose. Retry only when the source, method, environment, or hypothesis changes. An unchanged repeated failure becomes **manual-gap** with its impact and one exact next step. This step is complete when every component is verified or has an unavoidable, precisely documented user action; setup is not **ready** until all full-profile components verify.

## 3. Configure

Keep tool installations and services global, but MCP and agent-host configuration project-local. Before editing an existing host config, preserve its exact contents in an operation-owned temporary backup. Merge only the required MCP and skill settings, preserve unrelated fields, store no credentials, and parse every changed file.

On a parse or write failure, restore the affected file, preserve the primary error, and retry only with a changed correction. Release temporary backups after verification. Complete when every applicable host configuration parses and points at the installed components.

## 4. Verify

Verify skill discovery; every tool's version, help probe, or service health; MCP initialization and **tools/list**; and one harmless read-only MCP call where available. Verify that this workspace's MCP entries exist only in project-local configuration. Do not scan a target during setup, and do not treat any model provider as a prerequisite.

Route a missing or broken tool back to Install and an integration failure back to Configure. A user-only boundary becomes **manual-gap** with one exact action. Set **ready** only when the full profile works end to end. Leave installed components in place; the ledger supplies exact cleanup for a later user-requested removal.
