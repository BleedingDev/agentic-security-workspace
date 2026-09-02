# Project overview

Agentic Security Workspace gives coding agents a dedicated place to run security testing and reverse-engineering tools. It keeps those tools out of unrelated coding projects and gives each assessment a repeatable structure.

## Why it exists

Security tools often assume a person will sit in front of a GUI. Coding agents need commands or protocols they can call, inspect, and retry.

Nuclei already has a good command line. Burp and Ghidra need protocol bridges. Agent configuration also needs a home that will not affect ordinary coding work. This repository handles those differences through small tool skills and project-local Model Context Protocol configuration.

Burp, Ghidra, Nuclei, and the other tools still come from their maintainers. This project does not fork or bundle them. That choice is boring on purpose. A side project should not become a warehouse of stale installers and copied documentation.

~~~mermaid
flowchart LR
    O["Operator"] --> H["Local agent host"]
    H --> S["Project skills"]
    S --> C["CLI tools"]
    S --> M["Project-local MCP config"]
    M --> B["Burp"]
    M --> G["Ghidra"]
    C --> E["Per-target evidence"]
    B --> E
    G --> E
~~~

Tool binaries and long-running services stay global. The skills and agent-host configuration belong to this checkout. Each assessment keeps its scope, plan, evidence, and report with the target project.

## What local and sovereign mean

Local has a narrow meaning here. The tools run on the operator's machine, the agent configuration belongs to this checkout, and target evidence can stay in local files.

Sovereign means the operator chooses the agent host, model, tools, and network policy. The repository does not require a particular model provider. It also does not make a cloud model private. If the host sends prompts or tool results to a hosted model, that data leaves the machine.

Call a deployment offline only after using a local model, blocking unneeded network access, and checking the update and telemetry behavior of every installed tool. The repository helps organize that deployment. It cannot prove those controls for the operator.

## How setup works

The user invokes **setup-security-workspace** from the repository root. That invocation authorizes routine installation of the full supported tool set. The agent then:

1. Inspects the operating system, architecture, available package managers, existing tools, and agent host.
2. Installs missing components from current official sources. It keeps compatible existing installations.
3. Writes MCP settings only to project-local host configuration.
4. Verifies commands, services, MCP initialization, tool discovery, and harmless read-only calls.
5. Records ownership, versions, paths, evidence, failures, and exact removal instructions in **security-artifacts/setup.md**.

The agent pauses when a license, password prompt, GUI approval, driver, or device connection needs a person. It asks for one exact action and resumes after that action. A failed command only gets another attempt when the method, environment, source, or diagnosis changes.

The [setup skill](../.agents/skills/foundations/setup-security-workspace/SKILL.md) is the canonical workflow. Its [source catalog](../.agents/skills/foundations/setup-security-workspace/references/tool-sources.md) points to vendor documentation instead of copying install commands into this repository.

## Test record

The first development test ran on 2 September 2026 with macOS 26.7 on arm64.

| Check | Observed result |
|---|---|
| Skill package | All 24 skills validated. All 24 Claude compatibility links resolved in a clean public worktree. |
| Burp MCP | Initialization passed. The server listed 24 tools and completed a harmless Base64 encoding call. |
| Ghidra MCP | Initialization passed. The server listed 222 tools and completed a harmless instance-listing call. |
| Nuclei | The installed CLI reported version 3.11.1. |
| OpenCode | Both MCP servers connected from project-local configuration. |
| Codex | A trusted checkout loaded both project-local MCP servers. |
| Claude Code | Claude Code discovered both servers. First-use approval still required a person. |
| Target activity | Verification sent no scan and no target traffic. |

This result belongs to the named versions and date. Later tool releases may behave differently.

## Try it without scanning anything

Start an agent in a fresh clone and use this request:

~~~text
Run setup-security-workspace. Do not scan any target. Install the full profile,
then verify skill discovery, tool versions, MCP initialization, tools/list, and
one harmless read-only call per MCP server. Record the result and all removal
instructions in security-artifacts/setup.md.
~~~

The result is a local ledger. It distinguishes tools that already existed from tools setup installed. It also names every manual step and failed boundary.

## What the repository contains

- Small skills for web, API, Android, native binary, and source-code assessment.
- Focused tool guidance for Burp, Ghidra, Nuclei, Katana, MobSF, JADX, Apktool, ADB, Frida, capa, Semgrep, OSV-Scanner, and Trivy.
- A setup and assessment state model with explicit retry and failure rules.
- Pointers to upstream projects and established testing methods.

It does not contain commercial binaries, upstream source copies, model weights, credentials, target files, captured traffic, or assessment evidence. The Git ignore rules exclude local host configuration and the **security-artifacts/** directory.

## Limits

This is not an autonomous exploitation system. The skills require an explicit authorization boundary, bounded techniques, preserved evidence, and a finding review. Tool output starts as a candidate. It becomes a finding only after the agent traces cause and impact.

GUI and operating-system security prompts still need a person. Mobile runtime testing still needs an authorized emulator or device. Suspicious native code still needs an isolated environment before execution. Without those conditions, the agent records an untested gap instead of improvising.

Cross-platform setup is an agent procedure, not a matrix of checked-in shell scripts. The agent reads current vendor instructions for the detected system. This cuts maintenance work, but each new platform still needs real verification.

## Licensing

The MIT license covers the original files in this repository. It does not relicense Burp, Ghidra, MCP bridges, command-line tools, templates, or other dependencies. Those projects keep their own licenses and terms. The setup agent installs them from upstream sources. Users should review those terms before distribution or commercial use.
