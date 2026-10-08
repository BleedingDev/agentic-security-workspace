mod common;

use common::{assert_eq_normalized, render_pipeline_until, render_rule};
use wakaru_core::rules::RemoveVoid;

fn apply(input: &str) -> String {
    render_rule(input, RemoveVoid::new)
}

#[test]
fn transforms_void_zero_in_comparison() {
    // Reused from packages/unminify/src/transformations/__tests__/un-undefined.spec.ts
    let input = r#"
if(void 0 !== a) {
  console.log('a')
}
"#;
    let expected = r#"
if (undefined !== a) {
  console.log('a');
}
"#;
    let output = apply(input);
    assert_eq_normalized(&output, expected);
}

#[test]
fn transforms_void_numeric_literals() {
    // Reused from packages/unminify/src/transformations/__tests__/un-undefined.spec.ts
    let input = r#"
const a = void 0;
const b = void 99;
const c = void(0);
"#;
    let expected = r#"
const a = undefined;
const b = undefined;
const c = undefined;
"#;
    let output = apply(input);
    assert_eq_normalized(&output, expected);
}

#[test]
fn does_not_transform_void_under_delete() {
    let input = r#"
assert.sameValue(delete void 0, true);
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn does_not_transform_void_function_call() {
    // Reused from packages/unminify/src/transformations/__tests__/un-undefined.spec.ts
    // ArrowFunction rule converts the function expression to an arrow function.
    let input = r#"
const x = void function() {
  console.log('a');
  return void a();
};
"#;
    let expected = r#"
const x = void function() {
  console.log('a');
  return void a();
};
"#;

    let output = apply(input);
    assert_eq_normalized(&output, expected);
}

#[test]
fn does_not_transform_when_undefined_is_declared() {
    // Reused from packages/unminify/src/transformations/__tests__/un-undefined.spec.ts
    // VarDeclToLetConst converts `var undefined = 42` to `const` since it's never reassigned.
    let input = r#"
var undefined = 42;

console.log(void 0);

if (undefined !== a) {
  console.log('a', void 0);
}
"#;
    let expected = r#"
var undefined = 42;
console.log(undefined);
if (undefined !== a) {
  console.log('a', undefined);
}
"#;

    let output = apply(input);
    assert_eq_normalized(&output, expected);
}

#[test]
fn skips_module_with_direct_eval() {
    // Sloppy direct `eval` can declare `undefined` at runtime; the synthesized
    // global reference has no static proof it stays free, so the whole module
    // keeps `void 0` (dynamic-scope policy in docs/rewrite-assumptions.md).
    let input = r#"
function f(code) {
  eval(code);
  return void 0;
}
"#;
    let output = render_pipeline_until(input, "RemoveVoid");
    assert!(output.contains("void 0"), "{output}");
    assert!(!output.contains("undefined"), "{output}");
}

#[test]
fn skips_module_with_with_statement() {
    let input = r#"
function f(scope) {
  with (scope) {
    log(void 0);
  }
  return void 0;
}
"#;
    let output = render_pipeline_until(input, "RemoveVoid");
    assert_eq!(output.matches("void 0").count(), 2, "{output}");
    assert!(!output.contains("undefined"), "{output}");
}

#[test]
fn indirect_eval_does_not_block() {
    // `(0, eval)(code)` runs in the global scope and cannot add a binding to
    // this module's lexical scope.
    let input = r#"
function f(code) {
  (0, eval)(code);
  return void 0;
}
"#;
    let output = render_pipeline_until(input, "RemoveVoid");
    assert!(output.contains("return undefined"), "{output}");
}

#[test]
fn known_eval_source_blocks_only_when_it_mentions_undefined() {
    let unrelated = r#"
const crypto = eval("require('crypto')");
const a = void 0;
"#;
    let output = render_pipeline_until(unrelated, "RemoveVoid");
    assert!(output.contains("const a = undefined"), "{output}");

    let mentions = r#"
eval("var undefined = 1");
const a = void 0;
"#;
    let output = render_pipeline_until(mentions, "RemoveVoid");
    assert!(output.contains("void 0"), "{output}");
    assert!(!output.contains("const a = undefined"), "{output}");
}

#[test]
fn skips_module_with_non_variable_undefined_bindings() {
    // A named function expression, a class name, or an import local also
    // capture the printed `undefined`, not only variable-like bindings.
    for source in [
        "(function undefined() { log(void 0 === x); })();",
        "class undefined { static none() { return void 0; } }",
        "import undefined from 'm'; const a = void 0;",
    ] {
        let output = render_pipeline_until(source, "RemoveVoid");
        assert!(output.contains("void 0"), "{source}\n{output}");
        assert!(!output.contains("= undefined"), "{source}\n{output}");
    }
}
