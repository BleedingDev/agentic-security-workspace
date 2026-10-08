// Real two-stage producer: inline/external SWC async wrappers around a tslib generator.
const fs = require("node:fs");
const path = require("node:path");
const swc = require("@swc/core");
const ts = require("typescript");
const source = fs.readFileSync(path.join(__dirname, "source.js"), "utf8");
for (const externalHelpers of [false, true]) {
  const first = swc.transformSync(source, {
    filename: "input.js",
    jsc: { target: "es2015", parser: { syntax: "ecmascript" }, externalHelpers },
    module: { type: "es6" },
  }).code;
  const second = ts.transpileModule(first, {
    compilerOptions: { target: ts.ScriptTarget.ES5, module: ts.ModuleKind.CommonJS, importHelpers: true },
  }).outputText;
  fs.writeFileSync(path.join(__dirname, externalHelpers ? "external-generated.js" : "generated.js"), second);
  if (externalHelpers) fs.writeFileSync(path.join(__dirname, "external-es2015.js"), first);
}
