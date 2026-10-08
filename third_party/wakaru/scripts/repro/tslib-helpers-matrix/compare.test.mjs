import assert from "node:assert/strict";
import test from "node:test";
import { validateModuleRecovery } from "./compare.mjs";

const source = "export async function load(value) { return await value; }";
function matches(recovered, snippet = { source }) {
  return validateModuleRecovery({ snippet, recovered }).recovered;
}

test("module comparison accepts export relocation and minified bindings", () => {
  assert.ok(matches("export { f as load }; async function f(v) { return await v; }"));
  assert.ok(matches('import t from "tslib"; async function f(v) { return await v; } export { f as load };'));
  assert.ok(matches('import { __awaiter as a } from "tslib"; export async function load(v) { return await v; }'));
});

test("module comparison rejects incomplete helper recovery despite async syntax", () => {
  assert.equal(matches('import t from "tslib"; export async function load(v) { return t.__generator(this, v); }'), false);
  assert.equal(matches('import { __awaiter as a } from "tslib"; export async function load(v) { return await a(v); }'), false);
  assert.equal(matches('function helper() {} export async function load(v) { return await v; }'), false);
});

test("module comparison preserves public exports and other side effects", () => {
  assert.equal(matches("async function load(v) { return await v; }"), false);
  assert.equal(matches("export async function wrong(v) { return await v; }"), false);
  assert.equal(matches('import "./effect.js"; export async function load(v) { return await v; }'), false);
  assert.equal(matches('consume(); export async function load(v) { return await v; }'), false);
  assert.equal(matches("export async function load(v) { return await other; }"), false);
});

test("module comparison rejects leftover yield-star wrappers", () => {
  const snippet = { source: "export function* values(items) { yield* items; }" };
  assert.ok(matches('import t from "tslib"; export function* values(i) { yield* i; }', snippet));
  assert.equal(matches('import t from "tslib"; export function* values(i) { yield* t.__values(i); }', snippet), false);
});

test("module comparison accepts only explicit alternate bodies", () => {
  const snippet = {
    source: "export function rest(obj) { const {x, ...tail} = obj; return tail; }",
    acceptForms: ["export function rest({x, ...tail}) { return tail; }"],
  };
  assert.ok(matches("function r({x, ...t}) { return t; } export { r as rest };", snippet));
  assert.equal(matches("export function rest(obj) { return obj; }", snippet), false);
});

test("module comparison does not normalize away a provider import mismatch", () => {
  const snippet = { source: "import fn from './provider.js'; export function run() { return fn(); }" };
  assert.ok(matches('import p from "./provider.js"; function r() { return p(); } export { r as run };', snippet));
  assert.equal(matches('import p from "./wrong.js"; export function run() { return p(); }', snippet), false);
});

test("private names compare by class binding while public properties remain exact", () => {
  const snippet = { source: "export class Foo { #x = 1; #y = 2; read() { return this.#x - this.#y; } }" };
  assert.ok(matches("export class Foo { #a = 1; #b = 2; read() { return this.#a - this.#b; } }", snippet));
  assert.equal(matches("export class Foo { #a = 1; #b = 2; read() { return this.#b - this.#a; } }", snippet), false);
  assert.equal(matches("export class Foo { #a = 1; #b = 2; wrong() { return this.#a - this.#b; } }", snippet), false);
});

test("private-name comparison respects nested scopes and heritage evaluation", () => {
  const snippet = { source: "export class Outer { #x = 1; inner() { return class extends base(this.#x) { #y = 2; read(o) { return o.#x + this.#y; } }; } }" };
  assert.ok(matches("export class Outer { #a = 1; inner() { return class extends base(this.#a) { #b = 2; read(o) { return o.#a + this.#b; } }; } }", snippet));
  assert.equal(matches("export class Outer { #a = 1; inner() { return class extends base(this.#a) { #a = 2; read(o) { return o.#a + this.#a; } }; } }", snippet), false);
});

test("private-name comparison preserves brand checks and rejects unbound names", () => {
  const snippet = { source: "export class Foo { #x; has(o) { return #x in o; } }" };
  assert.ok(matches("export class Foo { #r; has(o) { return #r in o; } }", snippet));
  assert.equal(matches("export class Foo { #r; has(o) { return #__private0 in o; } }", snippet), false);
});

test("default class exports retain their spelling unless an alternate form is explicit", () => {
  const source = "export default class Foo { #x = 1; getX() { return this.#x; } }";
  const split = "class Foo { #x = 1; getX() { return this.#x; } } export default Foo;";
  assert.ok(matches(source, { source }));
  assert.equal(matches("export class Foo { #x = 1; getX() { return this.#x; } }", { source }), false);
  assert.equal(matches(split, { source }), false);
  assert.ok(matches(split, { source, acceptForms: [split] }));
  assert.equal(matches(split.replace("export default Foo", "export default other"), { source, acceptForms: [split] }), false);
});

test("class-expression recovery requires the direct initializer and original export", () => {
  const source = "const Foo = class { #x = 1; getX() { return this.#x; } setX(value) { this.#x = value; } }; export { Foo };";
  const recovered = "export const Foo = class { #x = 1; getX() { return this.#x; } setX(value) { this.#x = value; } };";
  const split = "export let Foo; Foo = class { #x = 1; getX() { return this.#x; } setX(value) { this.#x = value; } };";
  const snapshot = "let temp; temp = class { #x = 1; getX() { return this.#x; } setX(value) { this.#x = value; } }; export const Foo = temp;";
  const snippet = { source };
  assert.ok(matches(recovered, snippet));
  assert.equal(matches(recovered.replace("export ", ""), snippet), false);
  assert.equal(matches(recovered.replace("const Foo", "let Foo"), snippet), false);
  assert.equal(matches(split, snippet), false);
  assert.equal(matches(snapshot, snippet), false);
  assert.equal(matches(recovered + "Foo = other;", snippet), false);
  assert.equal(matches(recovered.replace("const Foo", "const Other"), snippet), false);
  assert.equal(matches(recovered.replace("this.#x", "get(this, map)"), snippet), false);
});
