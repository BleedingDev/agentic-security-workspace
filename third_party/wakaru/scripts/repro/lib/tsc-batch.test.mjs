import assert from "node:assert/strict";
import test from "node:test";
import { tscBatch } from "./runner.mjs";

// A source module is required to exercise importHelpers. Without export the
// compiler silently keeps helpers inline, which would hide namespace gaps.
const source = "export async function load(value) { return await value; }";
const options = { version: "5.9.3", target: "ES5" };

test("tsc profiles expose inline, namespace, and named helper delivery", async () => {
  const [inline, commonjs, esm] = await Promise.all([
    tscBatch([source], options),
    tscBatch([source], { ...options, module: "CommonJS", importHelpers: true }),
    tscBatch([source], { ...options, importHelpers: true }),
  ]);
  assert.match(inline.get(source), /var __awaiter/);
  assert.match(inline.get(source), /export function load/);
  assert.doesNotMatch(inline.get(source), /["']tslib["']/);
  assert.match(commonjs.get(source), /require\("tslib"\)/);
  assert.match(commonjs.get(source), /tslib_1\.__awaiter\(/);
  assert.match(commonjs.get(source), /tslib_1\.__generator\(/);
  assert.match(esm.get(source), /import \{ __awaiter, __generator \} from "tslib"/);
  assert.doesNotMatch(esm.get(source), /var __awaiter/);
});

test("tsc importHelpers does not externalize script helpers", async () => {
  const script = source.replace("export ", "");
  const output = (await tscBatch([script], { ...options, importHelpers: true })).get(script);
  assert.match(output, /var __awaiter/);
  assert.doesNotMatch(output, /["']tslib["']/);
});

test("tsc iteration and interop options expose their tslib calls", async () => {
  const sources = [
    "export function* values(items) { yield* items; }",
    "import value from './provider.js'; export default value;",
  ];
  const output = await tscBatch(sources, {
    ...options, module: "CommonJS", importHelpers: true,
    downlevelIteration: true, esModuleInterop: true,
  });
  assert.match(output.get(sources[0]), /tslib_1\.__values\(items\)/);
  assert.match(output.get(sources[1]), /tslib_1\.__importDefault\(/);
});
