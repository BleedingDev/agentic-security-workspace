// NODE_PATH=<tslib node_modules> WAKARU=<binary> node --experimental-vm-modules runtime.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { execFileSync } = require("node:child_process");
const tslib = require("tslib");
const binary = process.env.WAKARU;
assert(binary, "Set WAKARU to the binary under test");
const native = fs.readFileSync(path.join(__dirname, "source.ts"), "utf8");
const ordinary = "function Parent(value) { this.value = value; }";
const cases = [
  { name: "ordinary", setup: ordinary, lowered: 41, standard: 41 },
  { name: "native parent", setup: "class Parent { constructor(value) { this.value = value; } }", lowered: "TypeError", standard: 41 },
  { name: "null parent", setup: "var Parent = null;", lowered: undefined, standard: "TypeError" },
  { name: "overridden apply", setup: ordinary + " Parent.apply = function(self) { self.value = 99; };", lowered: 99, standard: 41 },
];
async function observe(code, setup, exported = "Child") {
  const context = vm.createContext({ exports: {}, require(name) { assert.equal(name, "tslib"); return tslib; } });
  vm.runInContext(setup, context);
  let Constructor;
  if (/(?:^|[;\n])\s*(?:import|export)\b/.test(code)) {
    const module = new vm.SourceTextModule(code, { context });
    await module.link(name => {
      assert.equal(name, "tslib");
      return new vm.SyntheticModule(Object.keys(tslib), function() {
        for (const [key, value] of Object.entries(tslib)) this.setExport(key, value);
      }, { context });
    });
    await module.evaluate();
    Constructor = module.namespace[exported];
  } else {
    vm.runInContext(code, context);
    Constructor = context.exports[exported];
  }
  try {
    const value = Constructor.make(41);
    assert.equal(Object.getPrototypeOf(value), Constructor.prototype, "factory must construct its own class");
    return value.value;
  } catch (error) { return error.name; }
}
(async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "wakaru-static-factory-"));
  function recover(code, level) {
    const input = path.join(temp, "input.js"), output = path.join(temp, "output.js");
    fs.writeFileSync(input, code);
    execFileSync(binary, [input, "--level", level, "-o", output, "--force"], { stdio: "pipe" });
    return fs.readFileSync(output, "utf8");
  }
  try {
    const files = fs.readdirSync(__dirname).filter(name => name.endsWith(".js"));
    assert.equal(files.length, 9);
    for (const file of files) {
      const lowered = fs.readFileSync(path.join(__dirname, file), "utf8");
      const minimal = recover(lowered, "minimal"), standard = recover(lowered, "standard");
      for (const test of cases) {
        assert.equal(await observe(lowered, test.setup), test.lowered, `${file}: ${test.name}, lowered`);
        assert.equal(await observe(minimal, test.setup), test.lowered, `${file}: ${test.name}, minimal`);
        assert.equal(await observe(standard, test.setup), test.standard, `${file}: ${test.name}, standard`);
        assert.equal(await observe(native, test.setup), test.standard, `${file}: ${test.name}, native`);
      }
    }
    // The inner constructor is a distinct binding from the reassigned outer variable.
    for (const inner of ["Child", "C"]) {
      const lowered = `import { __extends } from "tslib";
      var Child = (function(base) {
        __extends(${inner}, base);
        function ${inner}() { return base !== null && base.apply(this, arguments) || this; }
        ${inner}.make = function(value) { return new (${inner})(value); };
        return ${inner};
      })(Parent);
      var Saved = Child; Child = function Replacement() {}; export { Saved };`;
      for (const code of [lowered, recover(lowered, "minimal"), recover(lowered, "standard")]) {
        assert.equal(await observe(code, ordinary, "Saved"), 41, `outer reassignment: ${inner}`);
      }
    }
    console.log("36 compiler/profile runtime cases + 2 outer-binding reassignment cases passed");
  } finally { fs.rmSync(temp, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
