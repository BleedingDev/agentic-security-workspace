// Run with: NODE_PATH=<tslib 2.8.1 node_modules> WAKARU=<binary> node --experimental-vm-modules runtime.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { execFileSync } = require("node:child_process");
const tslib = require("tslib");
const binary = process.env.WAKARU;
assert(binary, "Set WAKARU to the binary under test");
const native = "export class Child extends Parent { value() { return super.value() + 1; } }";
const ordinary = "function Parent(x) { this.x = x; } Parent.prototype.value = function() { return this.x; };";
const cases = [
  { name: "ordinary function", setup: ordinary, lowered: 42, recovered: 42 },
  { name: "null", setup: "var Parent = null;", lowered: "constructed", recovered: "TypeError", constructOnly: true },
  { name: "native class", setup: "class Parent { constructor(x) { this.x = x; } value() { return this.x; } }", lowered: "TypeError", recovered: 42 },
  { name: "overridden apply", setup: ordinary + " Parent.apply = function(self) { self.x = 98; };", lowered: 99, recovered: 42 },
  { name: "overridden call", setup: ordinary + " Parent.prototype.value.call = function() { return 98; };", lowered: 99, recovered: 42 },
  { name: "replaced parent prototype", setup: ordinary, after: "Parent.prototype = { value() { return 98; } };", lowered: 99, recovered: 42 },
  { name: "changed child prototype chain", setup: ordinary, after: "Object.setPrototypeOf(Child.prototype, { value() { return 98; } });", lowered: 42, recovered: 99 },
];
async function observe(code, test) {
  const context = vm.createContext({ exports: {}, require(name) { assert.equal(name, "tslib"); return tslib; } });
  vm.runInContext(test.setup, context);
  let Child;
  if (/(?:^|[;\n])\s*(?:import|export)\b/.test(code)) {
    const module = new vm.SourceTextModule(code, { context });
    await module.link(name => {
      assert.equal(name, "tslib");
      return new vm.SyntheticModule(Object.keys(tslib), function () {
        for (const [key, value] of Object.entries(tslib)) this.setExport(key, value);
      }, { context });
    });
    await module.evaluate();
    Child = module.namespace.Child;
  } else {
    vm.runInContext(code, context);
    Child = context.exports.Child;
  }
  context.Child = Child;
  if (test.after) vm.runInContext(test.after, context);
  try {
    const child = new Child(41);
    return test.constructOnly ? "constructed" : child.value();
  } catch (error) { return error.name; }
}
(async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "wakaru-ts-inheritance-"));
  try {
    const files = fs.readdirSync(__dirname).filter(name => name.endsWith(".js"));
    assert.equal(files.length, 9);
    for (const file of files) {
      const input = path.join(__dirname, file);
      const lowered = fs.readFileSync(input, "utf8");
      const outputs = {};
      for (const level of ["minimal", "standard"]) {
        const out = path.join(temp, `${file}-${level}.js`);
        execFileSync(binary, [input, "--level", level, "-o", out], { stdio: "pipe" });
        outputs[level] = fs.readFileSync(out, "utf8");
      }
      for (const test of cases) {
        const label = `${file}: ${test.name}`;
        assert.equal(await observe(lowered, test), test.lowered, `${label} lowered`);
        assert.equal(await observe(outputs.minimal, test), test.lowered, `${label} minimal`);
        assert.equal(await observe(native, test), test.recovered, `${label} native source`);
        assert.equal(await observe(outputs.standard, test), test.recovered, `${label} standard`);
      }
      console.log(`${file}: ${cases.length} runtime cases passed (minimal = lowered; standard = native source)`);
    }
  } finally { fs.rmSync(temp, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
