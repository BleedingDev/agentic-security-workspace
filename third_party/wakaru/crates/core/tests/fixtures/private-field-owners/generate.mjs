// Run from the repository root: node crates/core/tests/fixtures/private-field-owners/generate.mjs
import { tscBatch } from "../../../../../scripts/repro/lib/runner.mjs";
import { writeFileSync } from "node:fs";
const sources = {
  default: "export default class Foo { #x = 1; getX() { return this.#x; } setX(value) { this.#x = value; } }",
  expression: "const Foo = class { #x = 1; getX() { return this.#x; } setX(value) { this.#x = value; } }; export { Foo };",
};
for (const [label, module] of [["esm", "ESNext"], ["cjs", "CommonJS"]]) {
  const results = await tscBatch(Object.values(sources), {
    version: "5.9.3", target: "ES2015", module, importHelpers: false,
  });
  for (const [name, source] of Object.entries(sources)) {
    const result = results.get(source);
    if (typeof result !== "string") throw new Error(JSON.stringify(result));
    writeFileSync(new URL(`${name}-${label}.js`, import.meta.url), result);
  }
}
