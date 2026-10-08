import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { runNodeBatch, runNodeBatchSync, runNodeJsonArraySync } from "./tool-process.mjs";

function toolDirectory(t) {
  const dir = mkdtempSync(join(tmpdir(), "wakaru-tool-process-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const pkg = join(dir, "node_modules", "local-tool");
  mkdirSync(pkg, { recursive: true });
  writeFileSync(join(pkg, "package.json"), JSON.stringify({ name: "local-tool", main: "index.cjs" }));
  writeFileSync(join(pkg, "index.cjs"), "exports.prefix = 'compiled:';");
  return dir;
}

test("overlapping batches with the same label keep their own program and options", async t => {
  const cwd = toolDirectory(t);
  const program = id => `
    import fs from 'node:fs';
    import { setTimeout } from 'node:timers/promises';
    import tool from 'local-tool';
    const sources = JSON.parse(fs.readFileSync(0, 'utf8'));
    fs.writeFileSync('${id}.ready', '');
    const deadline = Date.now() + 5000;
    while (!fs.existsSync('${1-id}.ready')) {
      if (Date.now() > deadline) throw new Error('other batch did not start');
      await setTimeout(5);
    }
    console.log(JSON.stringify(sources.map(s => ({ code: tool.prefix + '${id}:' + process.env.PROFILE + ':' + s }))));
  `;
  const outputs = await Promise.all([0, 1].map(id =>
    runNodeBatch(program(id), ["input"], { cwd, label: "same-producer", env: { PROFILE: `profile${id}` } }),
  ));
  assert.equal(outputs[0].get("input"), "compiled:0:profile0:input");
  assert.equal(outputs[1].get("input"), "compiled:1:profile1:input");
});

for (const format of ["module", "commonjs"]) {
  test(`${format} sync and async batches resolve packages from the tool directory`, async t => {
    const cwd = toolDirectory(t);
    const imports = format === "module"
      ? "import fs from 'node:fs'; import tool from 'local-tool';"
      : "const fs = require('node:fs'); const tool = require('local-tool');";
    const source = `${imports}
      const inputs = JSON.parse(fs.readFileSync(0, 'utf8'));
      console.log(JSON.stringify(inputs.map(s => ({code: tool.prefix + s}))));`;
    const options = { cwd, format, label: "local producer" };
    const input = ["first", "繁體中文"];
    const sync = runNodeBatchSync(source, input, options);
    const async = await runNodeBatch(source, input, options);
    assert.deepEqual([...async], [...sync]);
    assert.equal(sync.get(input[1]), "compiled:繁體中文");
  });
}

test("producer errors remain per-source failures", async () => {
  const source = "console.log(JSON.stringify([{error:'unsupported syntax'},{code:'ok'}]));";
  for (const result of [runNodeBatchSync(source, ["bad", "good"]), await runNodeBatch(source, ["bad", "good"])]) {
    assert.equal(result.get("bad").message, "unsupported syntax");
    assert.equal(result.get("good"), "ok");
  }
});

for (const [name, output, reason] of [
  ["empty", "", /empty stdout/],
  ["malformed", "[", /invalid JSON/],
  ["non-array", "{}", /expected an array/],
  ["short", "[]", /expected 1 row/],
  ["extra", '[{"code":"a"},{"code":"b"}]', /expected 1 row/],
  ["invalid row", "[null]", /row 0/],
  ["missing code", "[{}]", /row 0/],
]) {
  test(`${name} JSON includes the producer and process diagnostics`, async () => {
    const source = `process.stderr.write('tool diagnostic'); process.stdout.write(${JSON.stringify(output)});`;
    const options = { label: "test producer" };
    const check = error => {
      assert.match(error.message, /test producer/);
      assert.match(error.message, reason);
      assert.match(error.message, /exit=0/);
      assert.match(error.message, /stdout=\d+ bytes/);
      assert.match(error.message, /tool diagnostic/);
      return true;
    };
    assert.throws(() => runNodeBatchSync(source, ["input"], options), check);
    await assert.rejects(runNodeBatch(source, ["input"], options), check);
  });
}

test("nonzero exits retain bounded stderr and signal diagnostics", async () => {
  const source = "process.stderr.write('x'.repeat(5000)); process.exitCode = 7;";
  await assert.rejects(runNodeBatch(source, ["input"], { label: "broken tool" }), error => {
    assert.match(error.message, /broken tool.*exit=7.*signal=none/s);
    assert.ok(error.message.length < 2000);
    return true;
  });
});

test("an early exit with pending input is reported without an unhandled pipe error", async () => {
  await assert.rejects(
    runNodeBatch("process.exit(7);", ["x".repeat(1024 * 1024)], { label: "early exit" }),
    /early exit:.*exit=7/,
  );
});

test("JSON arrays with custom rows are available for validation tools", () => {
  assert.deepEqual(runNodeJsonArraySync('console.log(JSON.stringify([{parse_ok:true}]));', ["file"]), [{parse_ok:true}]);
});
