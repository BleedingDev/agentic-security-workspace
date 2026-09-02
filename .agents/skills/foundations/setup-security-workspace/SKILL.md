---
name: setup-security-workspace
description: "Set up this checkout for one security-testing goal when the user invokes setup-security-workspace."
---

# Setup Security Workspace

Run once per checkout before the first assessment. This is a durable agent protocol, not an installer or workflow engine. Record the current state and decisions in **security-artifacts/setup.md** so another session can resume without repeating completed work.

## 1. Inspect

Read **README.md** and **AGENTS.md**. Identify the OS, architecture, agent host, target type, and the smallest applicable assessment and tool-skill slice. Check project-local configuration, executable discovery, versions, service health, and required manual or licensed components.

Classify each component as **ready**, **missing**, **manual**, or **failed**. Preserve the command and observed error for failures. Complete when every component in the smallest slice has a status and next probe.

## 2. Propose

Show **component | status | next step**. Prefer an official package-manager command; otherwise link official vendor instructions. The user performs privileged, GUI, license, and package-installation steps.

Propose only project-local integration changes. Host trust and first-use approval remain explicit user steps. Set state to **awaiting-confirmation**. Revise this step until the user accepts the slice; complete when every proposed mutation is explicit and confirmed.

## 3. Configure

Before editing an existing host config, preserve its exact contents in an operation-owned temporary backup. Merge the confirmed MCP and skill settings while preserving unrelated fields and storing no credentials. Parse every changed file.

On a parse or write failure, restore the affected files, preserve the primary error, and return to Propose with a narrower correction. Complete when all confirmed files parse and temporary backups have been released.

## 4. Verify

Verify skill discovery, relevant tool versions or service health, MCP initialization and **tools/list**, and one harmless read-only tool call where available. Verify that this workspace's MCP entries exist only in project-local config.

On failure, record the failing boundary and diagnose once. Retry only after the configuration, environment, or failure hypothesis changes. A repeated unchanged failure becomes **manual-gap** with one exact next step; a configuration-caused failure returns to Configure; a changed requirement returns to Propose.

Complete at **ready** when the selected slice works end to end. Complete at **manual-gap** only when every unresolved component has its evidence, impact, and exact next step recorded.
