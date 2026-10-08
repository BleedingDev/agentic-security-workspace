# REA setup lane

Consult [REA installation and setup](https://github.com/morluto/rea/blob/main/docs/installation.md) and the published `rea-agents` package. Repository main may describe capabilities absent from the release. Resolve the published version, check its supported Node/npm runtime, and record the exact package version and integrity in the setup ledger. Reuse compatible existing runtimes and engines.

## Install and configure locally

Install the CLI through the supported npm method or use a package runner pinned to the resolved version. Verify its version before registration. Use an absolute discovered executable path when the host cannot inherit the shell PATH. Record the installation owner and exact removal command. A project-local wrapper may supply the pin and verified engine environment; keep generated wrappers ignored.

Install that package's bundled `reverse-engineer-anything` skill and references under `.agents/skills/reverse-engineer-anything/`. Preserve its license and metadata. Add the ignored Claude compatibility link to `../../../.agents/skills/reverse-engineer-anything`. Verify every relative reference and link resolves. Keep this upstream package copy local; the tracked `/rea` skill owns workspace integration guidance.

Upstream `rea setup` defaults to global client configuration and a global skill destination. Inspect its dry-run paths before applying it. Create the required project-local stdio registration instead when those paths escape the checkout. Merge a named `rea` entry alongside existing Burp and Ghidra entries using the current host's supported schema. The command runs the pinned package's `mcp` subcommand. Keep the resolved version, launcher, bundled workflow, and actual MCP catalog aligned.

For Codex, use `.codex/config.toml` with a sufficient startup timeout and a tool timeout covering the native provider's documented startup deadline. For Claude Code, use `.mcp.json`; for OpenCode, preserve the existing `mcp` or `mcp.servers` structure in `opencode.json`. Configure only applicable hosts. Parse changed files and preserve unrelated settings using the setup protocol's temporary backups.

## Reuse engines and classify limits

Verify the installed Ghidra version, native decompiler, and full JDK against the selected REA release. Put discovered absolute `GHIDRA_INSTALL_DIR` and, when needed, `JAVA_HOME` in the local launch environment. Select Ghidra explicitly when multiple native providers could accept the target. REA creates temporary headless projects; it does not attach to the existing GUI bridge's program. Keep both integrations.

Android inspection requires a compatible headless JADX and full JDK. Static JavaScript/Electron and managed assembly inspection do not need a native engine. Browser and runtime capture require task-specific endpoints, permissions, and execution scope. Linux-only firmware or crash tools require an appropriate Linux environment and their external dependencies. Record unsupported combinations as gaps; do not install a commercial provider merely to remove an informational doctor warning.

## Verify and maintain

Verify CLI startup and scoped engine readiness. Upstream `doctor --client codex` inspects global configuration; it does not validate this local registration. Inspect the actual checkout-local config, initialize its MCP command, list tools, and make a target-free `binary_session` call when advertised. Check server version and catalog against the installed workflow. Restart or reconnect the host when its live tools predate registration.

For end-to-end analysis checks, create small owned synthetic JavaScript and native fixtures in registered temporary storage. Verify the application graph and native pseudocode, assembly, and caller evidence without executing the sample. Preserve raw results and commands in the ignored setup evidence directory. Close sessions and remove owned fixtures, projects, and test outputs. A successful server handshake alone does not prove native analysis works.

Record every tested lane and remaining platform or runtime gap in the setup ledger. Keep setup-owned installations for use, with exact removal instructions. Upgrade the pinned registration and bundled workflow together, repeat relevant checks, and reconnect the host; never infer availability from repository main alone.
