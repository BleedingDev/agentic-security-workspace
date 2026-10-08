import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { Formatter } from "incur";
import { resolve } from "node:path";

import { CATALOG_IDENTITY } from "./catalogIdentity.js";
import { PROMPT_CONTRACTS } from "./contracts/promptContracts.js";
import { TOOL_CONTRACTS } from "./contracts/toolContracts.js";
import { PRODUCT_IDENTITY } from "./identity.js";
import { MCP_STARTUP_POLICY } from "./mcpStartupPolicy.js";

const OUTPUT_FORMATS = ["toon", "json", "yaml", "md", "jsonl"] as const;
type OutputFormat = (typeof OUTPUT_FORMATS)[number];

interface McpDoctorOptions {
  readonly command: string;
  readonly args: readonly string[];
  readonly cwd: string;
  readonly environment: Readonly<Record<string, string | undefined>>;
  readonly deadlineMs?: number;
}

interface InventoryComparison {
  readonly expected: number;
  readonly observed: number;
  readonly missing: readonly string[];
  readonly unexpected: readonly string[];
  readonly duplicates: readonly string[];
}

interface McpDoctorCheck {
  readonly name: string;
  readonly ok: boolean;
  readonly detail: string;
}

/** Diagnose the actual production stdio adapter with one startup-bounded SDK session. */
export const runProductionMcpDoctor = async (options: McpDoctorOptions) => {
  const deadline =
    Date.now() + (options.deadlineMs ?? MCP_STARTUP_POLICY.doctorDeadlineMs);
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new Error("Production MCP doctor deadline expired")),
    Math.max(1, deadline - Date.now()),
  );
  const stderr = captureStderr();
  const transport = new TrackedStdioClientTransport({
    command: options.command,
    args: [...options.args],
    cwd: options.cwd,
    env: targetFreeEnvironment(options.environment),
    stderr: "pipe",
  });
  transport.stderr?.on("data", stderr.append);
  const client = new Client({ name: "rea-mcp-doctor", version: "1" });
  try {
    await client.connect(
      transport,
      requestOptions(deadline, controller.signal),
    );
    return {
      ...(await inspectProductionMcpSession(
        client,
        deadline,
        controller.signal,
      )),
      diagnostics: stderr.value(),
    };
  } catch (cause: unknown) {
    return {
      healthy: false,
      adapter: "production-stdio" as const,
      server: null,
      inventory: null,
      identity: null,
      request_flow: { tool: "binary_session", ok: false },
      checks: [
        {
          name: "transport",
          ok: false,
          detail: errorMessage(cause),
        },
      ],
      diagnostics: stderr.value(),
    };
  } finally {
    clearTimeout(timer);
    await closeMcpChild(client, transport);
  }
};

const inspectProductionMcpSession = async (
  client: Client,
  deadline: number,
  signal: AbortSignal,
) => {
  const request = requestOptions(deadline, signal);
  const [tools, prompts, requestFlow] = await Promise.all([
    client.listTools(undefined, request),
    client.listPrompts(undefined, request),
    client.callTool({ name: "binary_session", arguments: {} }, request),
  ]);
  const inventory = {
    tools: compareInventory(
      TOOL_CONTRACTS.map(({ name }) => name),
      tools.tools.map(({ name }) => name),
    ),
    prompts: compareInventory(
      PROMPT_CONTRACTS.map(({ name }) => name),
      prompts.prompts.map(({ name }) => name),
    ),
  };
  const observedIdentity = parseIdentityToolResult(
    requestFlow.structuredContent,
  );
  const serverVersion = client.getServerVersion();
  const protocolVersion = client.getNegotiatedProtocolVersion();
  const checks: McpDoctorCheck[] = [
    {
      name: "initialize",
      ok:
        serverVersion?.name === PRODUCT_IDENTITY.mcpServerKey &&
        serverVersion.version === PRODUCT_IDENTITY.packageVersion &&
        protocolVersion !== undefined,
      detail: `${serverVersion?.name ?? "unknown"}@${serverVersion?.version ?? "unknown"}; protocol=${protocolVersion ?? "unknown"}`,
    },
    ...Object.entries(inventory).map(([name, comparison]) =>
      inventoryCheck(name, comparison),
    ),
    {
      name: "server-identity",
      ok:
        observedIdentity.packageVersion === PRODUCT_IDENTITY.packageVersion &&
        observedIdentity.catalogDigest ===
          CATALOG_IDENTITY.digests.combined_sha256,
      detail: `package=${observedIdentity.packageVersion ?? "missing"}; catalog=${observedIdentity.catalogDigest ?? "missing"}`,
    },
    {
      name: "request-flow",
      ok: requestFlow.isError !== true,
      detail: "binary_session target-free request",
    },
  ];
  return {
    healthy: checks.every(({ ok }) => ok),
    adapter: "production-stdio" as const,
    server: {
      name: serverVersion?.name ?? null,
      version: serverVersion?.version ?? null,
      protocol_version: protocolVersion ?? null,
    },
    inventory,
    identity: {
      expected_package_version: PRODUCT_IDENTITY.packageVersion,
      observed_package_version: observedIdentity.packageVersion,
      expected_catalog_digest: CATALOG_IDENTITY.digests.combined_sha256,
      observed_catalog_digest: observedIdentity.catalogDigest,
    },
    request_flow: {
      tool: "binary_session",
      ok: requestFlow.isError !== true,
    },
    checks,
  };
};

/**
 * Dispatcher entry for `rea mcp doctor`, kept separate from Incur MCP setup.
 * @public
 */
export const runProductionMcpDoctorCli = async (
  arguments_: readonly string[],
  input: { readonly dispatcherPath: string; readonly packageRoot: string },
): Promise<{ readonly output: string; readonly exitCode: number }> => {
  const parsed = parseOutputArguments(arguments_);
  if (!parsed.ok)
    return {
      output: `${Formatter.format(
        { code: "VALIDATION_ERROR", message: parsed.message },
        parsed.format,
      )}\n`,
      exitCode: 1,
    };
  const result = await runProductionMcpDoctor({
    command: process.execPath,
    args: [resolve(input.dispatcherPath), "mcp"],
    cwd: input.packageRoot,
    environment: process.env,
  });
  return {
    output: `${Formatter.format(result, parsed.format)}\n`,
    exitCode: result.healthy ? 0 : 1,
  };
};

/** Compare exact names while retaining actionable mismatch locations. */
export const compareInventory = (
  expectedNames: readonly string[],
  observedNames: readonly string[],
): InventoryComparison => {
  const expected = new Set(expectedNames);
  const observed = new Set(observedNames);
  return {
    expected: expectedNames.length,
    observed: observedNames.length,
    missing: [...expected].filter((name) => !observed.has(name)).sort(),
    unexpected: [...observed].filter((name) => !expected.has(name)).sort(),
    duplicates: [...observed]
      .filter(
        (name) =>
          observedNames.filter((candidate) => candidate === name).length > 1,
      )
      .sort(),
  };
};

const inventoryCheck = (
  name: string,
  comparison: InventoryComparison,
): McpDoctorCheck => ({
  name: `inventory:${name}`,
  ok:
    comparison.expected === comparison.observed &&
    comparison.missing.length === 0 &&
    comparison.unexpected.length === 0 &&
    comparison.duplicates.length === 0,
  detail: `expected=${String(comparison.expected)} observed=${String(comparison.observed)} missing=${comparison.missing.join(",") || "none"} unexpected=${comparison.unexpected.join(",") || "none"} duplicates=${comparison.duplicates.join(",") || "none"}`,
});

const parseIdentityToolResult = (
  input: unknown,
): {
  readonly packageVersion: string | null;
  readonly catalogDigest: string | null;
} => {
  if (!isRecord(input) || !isRecord(input.result))
    return { packageVersion: null, catalogDigest: null };
  const identity = input.result.server_identity;
  if (!isRecord(identity)) return { packageVersion: null, catalogDigest: null };
  const packageIdentity = identity.package;
  const catalog = identity.catalog;
  const digests = isRecord(catalog) ? catalog.digests : undefined;
  return {
    packageVersion:
      isRecord(packageIdentity) && typeof packageIdentity.version === "string"
        ? packageIdentity.version
        : null,
    catalogDigest:
      isRecord(digests) && typeof digests.combined_sha256 === "string"
        ? digests.combined_sha256
        : null,
  };
};

const parseOutputArguments = (
  arguments_: readonly string[],
):
  | { readonly ok: true; readonly format: OutputFormat }
  | {
      readonly ok: false;
      readonly format: OutputFormat;
      readonly message: string;
    } => {
  let format: OutputFormat = "toon";
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === "--json") {
      format = "json";
      continue;
    }
    const inline = argument?.startsWith("--format=")
      ? argument.slice("--format=".length)
      : undefined;
    if (argument === "--format" || inline !== undefined) {
      const candidate = inline ?? arguments_[index + 1];
      if (inline === undefined) index += 1;
      if (!isOutputFormat(candidate))
        return {
          ok: false,
          format,
          message: `Invalid output format: ${candidate ?? "missing"}`,
        };
      format = candidate;
      continue;
    }
    if (argument === "--full-output") continue;
    return {
      ok: false,
      format,
      message: `Unknown mcp doctor option: ${argument ?? "missing"}`,
    };
  }
  return { ok: true, format };
};

const isOutputFormat = (value: string | undefined): value is OutputFormat =>
  OUTPUT_FORMATS.some((candidate) => candidate === value);

const requestOptions = (deadline: number, signal: AbortSignal) => ({
  signal,
  timeout: Math.max(1, deadline - Date.now()),
});

const targetFreeEnvironment = (
  environment: Readonly<Record<string, string | undefined>>,
): Record<string, string> => {
  const result = Object.fromEntries(
    Object.entries(environment).filter(
      (entry): entry is [string, string] => entry[1] !== undefined,
    ),
  );
  for (const name of [
    "HOPPER_TARGET_PATH",
    "HOPPER_TARGET_KIND",
    "HOPPER_LOADER_ARGS_JSON",
  ])
    delete result[name];
  return result;
};

const captureStderr = () => {
  const chunks: Buffer[] = [];
  let bytes = 0;
  return {
    append(chunk: unknown) {
      const buffer = Buffer.isBuffer(chunk)
        ? chunk
        : Buffer.from(String(chunk), "utf8");
      bytes += buffer.byteLength;
      chunks.push(Buffer.from(buffer));
    },
    value: () => ({
      stderr: Buffer.concat(chunks, bytes).toString("utf8"),
      stderr_bytes: bytes,
    }),
  };
};

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const errorMessage = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

class TrackedStdioClientTransport extends StdioClientTransport {
  spawnedPid: number | null = null;

  override async start(): Promise<void> {
    await super.start();
    this.spawnedPid = this.pid;
  }
}

const closeMcpChild = async (
  client: Client,
  transport: TrackedStdioClientTransport,
): Promise<void> => {
  const pid = transport.spawnedPid ?? transport.pid;
  await Promise.allSettled([client.close()]);
  await Promise.allSettled([transport.close()]);
  if (pid === null || !(await processExists(pid))) return;
  await terminateProcess(pid, "SIGTERM");
  if (await waitForProcessExit(pid, 1_000)) return;
  await terminateProcess(pid, "SIGKILL");
  await waitForProcessExit(pid, 1_000);
};

const terminateProcess = async (
  pid: number,
  signal: NodeJS.Signals,
): Promise<void> => {
  try {
    process.kill(pid, signal);
  } catch (cause: unknown) {
    if (!(cause instanceof Error && "code" in cause && cause.code === "ESRCH"))
      throw cause;
  }
};

const waitForProcessExit = async (
  pid: number,
  timeoutMs: number,
): Promise<boolean> => {
  const deadline = Date.now() + timeoutMs;
  do {
    if (!(await processExists(pid))) return true;
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
  } while (Date.now() < deadline);
  return !(await processExists(pid));
};

const processExists = async (pid: number): Promise<boolean> => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (cause: unknown) {
    if (cause instanceof Error && "code" in cause && cause.code === "ESRCH")
      return false;
    throw cause;
  }
};
