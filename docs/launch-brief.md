# Launch brief

This is the source for posts, talks, and release notes about Agentic Security Workspace. Read the [README](../README.md) for operating instructions. Check the dated test record below before making claims about compatibility.

## The short version

Security tools often assume a person will sit in front of a GUI. Coding agents need commands or protocols they can call, inspect, and retry. Agentic Security Workspace gives them those interfaces and the instructions for using them.

The repository gives a coding agent a project-local set of security skills. Its setup skill installs the tools, connects supported Model Context Protocol servers, checks the result, and records how to remove everything it added. Burp, Ghidra, Nuclei, and the other tools still come from their maintainers. This project does not fork or bundle them.

That last choice is boring on purpose. A side project should not become a warehouse of stale installers and copied documentation.

## Why I built it

I wanted an agent to inspect web traffic in Burp, trace native code in Ghidra, and run narrow Nuclei checks without constant copy and paste. Nuclei already had a good command line. Burp and Ghidra needed protocol bridges. Agent configuration also needed a home that would not affect unrelated coding work.

The repository puts the guidance and host configuration in one dedicated checkout. Tool binaries and long-running services stay global. Each assessment keeps its scope, plan, evidence, and report with the target project.

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

## What local and sovereign mean here

Local has a narrow meaning. The tools run on the operator's machine, the agent configuration belongs to this checkout, and target evidence can stay in local files.

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

## What we tested

This development snapshot records tests run on 2 September 2026 with macOS 26.7 on arm64.

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

This is a dated result, not a promise that later tool releases will behave the same way. Rerun setup and update this table before publishing a new compatibility claim.

## A harmless demonstration

Start the agent in a fresh clone and give it this request:

~~~text
Run setup-security-workspace. Do not scan any target. Install the full profile,
then verify skill discovery, tool versions, MCP initialization, tools/list, and
one harmless read-only call per MCP server. Record the result and all removal
instructions in security-artifacts/setup.md.
~~~

The useful result is the ledger. It should distinguish tools that already existed from tools setup installed. It should also name every manual step and failed boundary. A polished terminal recording is less valuable if it hides those details.

## What the repository publishes

- Small skills for web, API, Android, native binary, and source-code assessment.
- Focused tool guidance for Burp, Ghidra, Nuclei, Katana, MobSF, JADX, Apktool, ADB, Frida, capa, Semgrep, OSV-Scanner, and Trivy.
- A setup and assessment state model with explicit retry and failure rules.
- Pointers to upstream projects and established testing methods.

It does not publish commercial binaries, upstream source copies, model weights, credentials, target files, captured traffic, or assessment evidence. The Git ignore rules exclude local host configuration and the **security-artifacts/** directory.

## Limits worth saying out loud

This is not an autonomous exploitation system. The skills require an explicit authorization boundary, bounded techniques, preserved evidence, and a finding review. Tool output starts as a candidate. It becomes a finding only after the agent traces cause and impact.

GUI and operating-system security prompts still need a person. Mobile runtime testing still needs an authorized emulator or device. Suspicious native code still needs an isolated environment before execution. Without those conditions, the agent records an untested gap instead of improvising.

Cross-platform setup is an agent procedure, not a matrix of checked-in shell scripts. The agent reads current vendor instructions for the detected system. That cuts maintenance work, but it means each new platform needs real verification.

## Licensing

The MIT license covers the original files in this repository. It does not relicense Burp, Ghidra, MCP bridges, command-line tools, templates, or other dependencies. Those projects keep their own licenses and terms. The setup agent installs them from upstream sources, and users should review those terms before distribution or commercial use.

## Claims a post can make

- The project gives coding agents a repeatable, project-scoped security workflow.
- The setup skill installs supported tools automatically and records how to remove what it added.
- Burp and Ghidra become callable through MCP integrations. CLI tools remain directly callable.
- Local configuration and target evidence stay out of the public repository by default.
- The dated macOS test above passed without scanning a target.

Do not claim that the project is fully offline, works on every operating system without intervention, replaces a security tester, guarantees vulnerability discovery, or ships the upstream tools. Evidence in this repository does not support those claims.

## A clean story for a post

Start with the real annoyance. The agent could reason about a target, but it could not click through security GUIs or retain a disciplined assessment state.

Then explain the design. Keep tools global, keep agent guidance local to one checkout, connect GUI tools through MCP, and use command-line tools directly. Let a setup skill choose current vendor-supported installation methods instead of maintaining another installer.

Show the proof. Run the harmless demonstration, open the ledger, and show **tools/list** for Burp and Ghidra. Say which platform and date you tested.

End with the honest sovereignty point. Local tools give the operator control over execution and evidence. Data privacy still depends on the selected model, agent host, network policy, and tool configuration. That distinction matters more than the word "local" in a headline.

## Before publishing

- Rerun the harmless demonstration on the platform named in the post.
- Update the dated test table with observed results.
- Check upstream links and licenses.
- Remove usernames, machine paths, credentials, target data, and captured traffic.
- Separate measured facts from plans and opinions.
