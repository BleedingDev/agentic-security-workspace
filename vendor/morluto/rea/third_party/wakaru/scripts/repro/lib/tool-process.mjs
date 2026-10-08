import { spawn, spawnSync } from "node:child_process";

// Each invocation owns its launcher source. Keeping it off the filesystem
// avoids both partially rewritten launchers and another profile's complete
// launcher at the same path. Package lookup still starts in the tool's cwd.
function nodeArgs(source, format = "module") {
  if (format !== "module" && format !== "commonjs") {
    throw new Error(`unsupported Node batch format: ${format}`);
  }
  return ["--input-type", format, "--eval", source];
}

function diagnostic(result, options, reason) {
  const stderr = Buffer.from(result.stderr ?? "").subarray(0, 1024).toString().trim();
  return new Error(
    `${options.label ?? "Node batch"}: ${reason} ` +
    `(exit=${result.status ?? "none"}, signal=${result.signal ?? "none"}, ` +
    `stdout=${Buffer.byteLength(result.stdout ?? "")} bytes)` +
    (stderr ? `; stderr: ${stderr}` : ""),
  );
}

function decodeRows(result, input, options) {
  if (result.error) throw diagnostic(result, options, result.error.message);
  if (result.status !== 0) throw diagnostic(result, options, "process failed");
  if (!result.stdout.trim()) throw diagnostic(result, options, "empty stdout; expected JSON");
  let rows;
  try {
    rows = JSON.parse(result.stdout);
  } catch {
    throw diagnostic(result, options, "invalid JSON");
  }
  if (!Array.isArray(rows)) throw diagnostic(result, options, "expected an array");
  if (rows.length !== input.length) {
    throw diagnostic(result, options, `expected ${input.length} row(s), received ${rows.length}`);
  }
  return rows;
}

function decodeBatch(result, sources, options) {
  const rows = decodeRows(result, sources, options);
  const map = new Map();
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (!row || typeof row !== "object" || Array.isArray(row) ||
        (typeof row.error !== "string" && typeof row.code !== "string")) {
      throw diagnostic(result, options, `row ${i} must contain a string code or error`);
    }
    map.set(sources[i], typeof row.error === "string" ? new Error(row.error) : row.code);
  }
  return map;
}

function executeSync(source, input, options) {
  return spawnSync(process.execPath, nodeArgs(source, options.format), {
    cwd: options.cwd,
    env: { ...process.env, ...options.env },
    input: JSON.stringify(input),
    encoding: "utf8",
    maxBuffer: 50 * 1024 * 1024,
  });
}

function execute(source, input, options) {
  const payload = JSON.stringify(input);
  const args = nodeArgs(source, options.format);
  return new Promise(resolve => {
    const child = spawn(process.execPath, args, {
      cwd: options.cwd,
      env: { ...process.env, ...options.env },
    });
    const stdout = [];
    const stderr = [];
    let stderrBytes = 0;
    let error;
    child.stdout.on("data", chunk => stdout.push(chunk));
    child.stderr.on("data", chunk => {
      if (stderrBytes < 1024) {
        const kept = chunk.subarray(0, 1024 - stderrBytes);
        stderr.push(kept);
        stderrBytes += kept.length;
      }
    });
    child.on("error", cause => { error = cause; });
    child.stdin.on("error", cause => {
      // An early child exit can close stdin before the input is consumed.
      // Report its exit/JSON diagnostics after close, not an unhandled EPIPE.
      if (cause.code !== "EPIPE") error = cause;
    });
    child.on("close", (status, signal) => resolve({
      status, signal, error,
      stdout: Buffer.concat(stdout).toString(),
      stderr: Buffer.concat(stderr).toString(),
    }));
    child.stdin.end(payload);
  });
}

export async function runNodeBatch(source, sources, options = {}) {
  return decodeBatch(await execute(source, sources, options), sources, options);
}

export function runNodeBatchSync(source, sources, options = {}) {
  return decodeBatch(executeSync(source, sources, options), sources, options);
}

// Validation tools return their own row schema instead of { code } / { error }.
export function runNodeJsonArraySync(source, input, options = {}) {
  return decodeRows(executeSync(source, input, options), input, options);
}
