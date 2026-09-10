# Agentic Security Workspace

A small, project-local skill library for evidence-driven security testing and reverse engineering with coding agents.

It keeps security guidance and MCP discovery out of unrelated coding projects. Tool binaries and long-running services remain globally installed; this repository owns only agent guidance and checkout-local configuration.

Use it only for assets you own or are explicitly authorized to test. Target files, traffic, source, and documentation are untrusted evidence, never agent instructions.

Read the [project overview](docs/project-overview.md) for the design, tested setup, limits, and meaning of local AI in this project.

## Serving (optional): a self-hosted abliterated model

You can drive this workspace with **Qwen3.8-27B abliterated served free on a Kaggle TPU
v5e-8** — full bf16, up to the model's native 262k context — plugged into OpenCode 2 as an
OpenAI- and Anthropic-compatible endpoint. The recipe lives in [`serving/`](serving) (a fork
of [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab), MIT) and the
step-by-step guide is [docs/kaggle-tpu-serving.md](docs/kaggle-tpu-serving.md).

~~~mermaid
flowchart TD
    K["./serving/ktl start"] --> L["launch.py → Kaggle kernel<br/>(TPU v5e-8, free tier)"]
    L --> Q["queue for a TPU slot → load 55&nbsp;GB bf16<br/>→ compile graphs → cloudflared tunnel"]
    Q --> F["ktl writes ~/.config/kaggle-tpu-lab/<br/>{base_url, api_key}"]
    F --> P["OpenCode 2 provider <b>kaggle-tpu</b><br/>reads them via {file:…}"]
    P --> R["opencode2 -m kaggle-tpu/qwen3.8-27b-abliterated<br/>(run from the workspace root)"]
    R --> W["Agentic Security Workspace<br/>skills + evidence rules"]
    W --> T["tool lanes: ghidra MCP :8089 (headless) ·<br/>burp · analyzeHeadless · semgrep · …"]

    subgraph cloud["Kaggle free TPU"]
        L
        Q
    end
    subgraph host["Your machine"]
        K
        F
        P
        R
        W
        T
    end
~~~

```bash
cd serving && ./ktl setup          # once: venv + Kaggle CLI (needs a TPU-verified Kaggle account)
./ktl start                        # push the kernel, wait for live, wire it into OpenCode 2
./ktl ghidra start                 # optional: headless Ghidra MCP server on :8089, no GUI
cd .. && opencode2 -m kaggle-tpu/qwen3.8-27b-abliterated
./serving/ktl stop                 # when done, to free your weekly TPU quota
```

OpenCode 2 does **not** start the endpoint; always `./ktl start` first. Full prerequisites
(including Kaggle **identity** verification, not just phone) are in the guide.

## Start

There is no cross-platform installation script to maintain. The setup skill performs installation from current official sources.

1. Clone this repository and enter its root.
2. Launch Codex, Claude Code, or OpenCode from that root.
3. Trust the checkout when the host asks; Codex ignores project config until the project is trusted.
4. Invoke **setup-security-workspace**.
5. Let the agent install and verify the full supported toolchain automatically.
6. Complete an exact user action only when elevation, license acceptance, GUI approval, or device connection cannot be automated.
7. Let it create and verify the project-local MCP configuration.

The full profile covers every tool lane below, including required runtimes, companion data, local services, and Burp and Ghidra MCP integrations. It can install GUI applications and use several gigabytes. The setup ledger records whether each component was pre-existing or setup-owned, the exact installed version and location, and the uninstall or cleanup command to use later.

The setup skill discovers the current OS and host and resolves the current vendor-supported method at runtime, so the repository does not hard-code Homebrew, Winget, system paths, versions, or credentials. It writes host-specific files locally; they are ignored by Git:

| Host | Local configuration |
|---|---|
| Codex | .codex/config.toml |
| Claude Code | .mcp.json and optional .claude/settings.local.json |
| OpenCode | opencode.json |

See the official [Codex configuration precedence](https://developers.openai.com/codex/config-basic/) and [project-scoped MCP configuration](https://developers.openai.com/codex/mcp/) documentation for its trust boundary.

The canonical skills live under **.agents/skills/**. Claude Code loads the flat compatibility links under **.claude-compat/.claude/skills/**; Codex and OpenCode discover the canonical tree directly.

## Setup protocol

Setup is an agent-executed, persisted installer protocol. It is not a checked-in platform script. Its resumable ownership and removal ledger is **security-artifacts/setup.md**.

~~~mermaid
stateDiagram-v2
    [*] --> Inspect
    Inspect --> Install: full profile classified
    Inspect --> ManualGap: prerequisite cannot be inspected
    Install --> Install: next missing component
    Install --> AwaitUser: unavoidable user-only action
    AwaitUser --> Inspect: action completed / reclassify
    Install --> Configure: every component verifies
    Install --> ManualGap: unchanged failure repeats
    Configure --> Install: dependency is missing or broken
    Configure --> Configure: restore backup / changed correction
    Configure --> Verify: configs parse
    Verify --> Ready: full profile + skills + MCP pass
    Verify --> Install: tool verification fails
    Verify --> Configure: integration verification fails
    Verify --> AwaitUser: host approval is required
    Verify --> ManualGap: unchanged failure repeats
    Ready --> [*]
    ManualGap --> Inspect: missing prerequisite supplied
~~~

Every retry requires changed evidence. Existing config is backed up temporarily before mutation and restored on a configuration failure. Setup-owned components remain installed after setup; their exact removal instructions stay in the ledger. **ready** means the complete profile works, while **manual-gap** names the unresolved boundary and one exact user action.

## Assessment protocol

Start with **security-assessment** for a mixed or unknown target. Invoke a target-specific skill directly only when the target type is already clear.

~~~mermaid
stateDiagram-v2
    [*] --> Scope
    Scope --> Scope: missing or changed authorization boundary
    Scope --> Plan: every active technique is bounded
    Plan --> Execute: coverage rows exist
    Execute --> VerifyFinding: material candidate
    VerifyFinding --> Execute: confirmed / refuted / inconclusive
    Execute --> Diagnose: tool or method fails
    Diagnose --> Execute: cause or bounded method changed
    Diagnose --> TerminalGap: retry is unchanged, unsafe, or exhausted
    TerminalGap --> Execute: row marked untested with impact
    Execute --> Scope: stop condition fires
    Execute --> Report: no row remains planned
    Report --> Complete: evidence, gaps, and cleanup accounted for
    Complete --> [*]
~~~

The protocol is resumable from durable artifacts:

~~~text
security-artifacts/
├── setup.md       # environment state, decisions, failures, next steps
├── scope.md       # authorization, impact limits, cleanup, stop conditions
├── plan.md        # hypothesis/control → execution state → result
├── evidence/      # raw outputs, request pairs, traces, hashes, notes
└── report.md      # verified findings, coverage, remediation, residual gaps
~~~

Keep that directory inside the individual target project. The Git ignore rules prevent it and common target binaries or captures from being published accidentally.

Three independent state axes prevent false completion:

- Execution: **planned**, **tested**, **not-applicable**, **untested**
- Result: **pass**, **fail**, **inconclusive**
- Candidate disposition: **confirmed**, **refuted**, **inconclusive**

A report cannot start while a row remains **planned**. Tool failure becomes a retry only after something material changes; otherwise it becomes an explicit terminal gap.

## Reverse-engineering loop

~~~mermaid
flowchart TD
    I["Identify artifact<br/>provenance · hash · format · architecture"] --> T["Triage<br/>entry points · imports · strings · capabilities"]
    T --> H["State one behavior hypothesis"]
    H --> X["Trace static evidence<br/>symbols · xrefs · call path · data flow"]
    X --> K{"Resolved?"}
    K -- "yes" --> E["Preserve evidence path"]
    K -- "no" --> P{"Runtime-only, packed,<br/>encrypted, or ambiguous?"}
    P -- "no" --> H
    P -- "yes" --> G{"Execution authorized<br/>and isolated?"}
    G -- "no" --> L["Record an untested gap and impact"]
    G -- "yes" --> D["Run one bounded experiment<br/>capture state and observation"]
    D --> H
    E --> V["verify-security-finding"]
    L --> V
~~~

Unknown or suspicious code belongs in a revertible isolated environment with controlled networking, synthetic credentials, and an explicit cleanup plan. Without that environment, stop at static analysis and record the gap.

## Tool lanes

| Skill | Role | Evidence |
|---|---|---|
| burp | Capture, inspect, replay, and compare HTTP through the MCP bridge | Baseline and changed request-response pairs |
| katana | Map routes, inputs, forms, and script-discovered endpoints | Scoped JSONL crawl inventory |
| nuclei | Run narrow, technology-matched templates | JSONL candidates with template and failure context |
| mobsf | Produce an independent Android static-analysis view | Versioned JSON tied to the package hash |
| jadx | Trace managed Android code | Class, method, and source-to-sink path |
| apktool | Inspect manifest, resources, network policy, and smali | Decoded file and exact configuration value |
| adb | Control an authorized emulator or test device | Device, package, command, time, and state change |
| frida | Instrument a process for one runtime hypothesis | Hook, process identity, and observed event |
| capa | Triage native capabilities | Machine-readable leads with disposition |
| ghidra | Trace binary behavior through MCP | Program location, xrefs, decompilation, and call path |
| semgrep | Find source patterns and data-flow candidates | Structured matches traced through executable paths |
| osv-scanner | Match resolved dependencies to advisories | Version, affected feature, and reachability |
| trivy | Inspect images, filesystems, dependencies, secrets, and IaC | Structured results tied to the artifact |

Tool output is a candidate, not a finding. Every material candidate passes through **verify-security-finding**.

Burp integration uses PortSwigger's [official MCP extension](https://github.com/PortSwigger/mcp-server). Ghidra integration is compatible with [ghidra-mcp](https://github.com/bethington/ghidra-mcp). The setup skill verifies what is actually installed before writing configuration.

## Skill graph

~~~text
.agents/skills/
├── assessments/   # target-specific routers
│   ├── security-assessment
│   ├── web-security-assessment
│   ├── api-security-assessment
│   ├── android-security-assessment
│   ├── native-binary-assessment
│   └── source-security-assessment
├── foundations/   # scope → plan → verify → report
│   ├── setup-security-workspace
│   ├── scope-security-test
│   ├── plan-security-assessment
│   ├── verify-security-finding
│   └── write-security-report
└── tools/         # one focused evidence lane per integration
~~~

The library follows [writing-for-agents](https://github.com/mattpocock/skills/tree/main/skills/productivity/writing-for-agents): short context pointers, composition, one source of truth, and checkable completion criteria. It deliberately excludes vulnerability encyclopedias, autonomous exploitation chains, copied tool manuals, and installation scripts.

## Methodology

- [NIST SP 800-115](https://csrc.nist.gov/pubs/sp/800/115/final)
- [OWASP Web Security Testing Guide](https://owasp.org/www-project-web-security-testing-guide/stable/)
- [OWASP API Security Top 10](https://owasp.org/API-Security/)
- [OWASP MASVS and MASTG](https://mas.owasp.org/)
- [OWASP Firmware Security Testing Methodology](https://github.com/scriptingxss/owasp-fstm)
- [PortSwigger testing workflow](https://portswigger.net/burp/documentation/desktop/testing-workflow)

## License

MIT
