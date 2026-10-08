mod common;

use common::{assert_eq_normalized, render_rule};
use wakaru_core::rules::UnBuiltinAliases;

fn apply(input: &str) -> String {
    render_rule(input, UnBuiltinAliases::new)
}

#[test]
fn inlines_module_var_builtin_member_aliases() {
    let input = r#"
var e = Object.freeze;
var r = Object.defineProperty;
use(e(r(strings, "raw", { value: e(raws) })));
"#;
    let expected = r#"
use(Object.freeze(Object.defineProperty(strings, "raw", {
    value: Object.freeze(raws)
})));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, expected);
}

#[test]
fn inlines_module_const_builtin_member_aliases() {
    let input = r#"
const e = Object.freeze;
use(e(value));
"#;
    let expected = r#"
use(Object.freeze(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, expected);
}

#[test]
fn public_exported_builtin_alias_remains_declared() {
    let input = r#"
var defineProperty = Object.defineProperty;
export { defineProperty };
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_var_alias_used_before_initializer() {
    let input = r#"
use(e);
var e = Object.freeze;
use(e(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_reassigned_var_alias() {
    let input = r#"
var e = Object.freeze;
e = other;
use(e(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_var_alias_when_direct_eval_can_observe_binding() {
    let input = r#"
var e = Object.freeze;
eval("e");
use(e(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_alias_from_local_builtin_shadow() {
    let input = r#"
const Object = fake;
var e = Object.freeze;
use(e(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_var_alias_redeclared_with_non_alias_init() {
    let input = r#"
var e = Object.freeze;
use(e(value));
var e = getPolyfill();
use2(e);
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_var_alias_redeclaring_non_alias_binding() {
    let input = r#"
var e = getPolyfill();
use(e);
var e = Object.freeze;
use2(e(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_var_alias_mutated_by_update_expression() {
    let input = r#"
var e = Object.freeze;
e++;
use(e(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn preserves_var_alias_removed_with_delete() {
    let input = r#"
var e = Object.freeze;
use(delete e);
use2(e(value));
"#;
    let output = apply(input);
    assert_eq_normalized(&output, input);
}

#[test]
fn alias_named_by_export_specifier_remains_declared() {
    // `export { o }` can only name a binding, so the alias declaration has to
    // stay even though every expression use could be inlined.
    let input = r#"
var o = Object.create;
var d = Object.defineProperty;
function f(q) {
    return d(o(q), "x", { value: 1 });
}
export { o, f };
"#;
    let expected = r#"
var o = Object.create;
function f(q) {
    return Object.defineProperty(o(q), "x", { value: 1 });
}
export { o, f };
"#;
    assert_eq_normalized(&apply(input), expected);
}

#[test]
fn alias_call_inside_a_computed_object_key_is_inlined() {
    let input = r#"
const o = Object.keys;
const m = { [o(x)[0]]: 1 };
o(y);
export { m };
"#;
    let output = apply(input);
    assert!(output.contains("[Object.keys(x)[0]]: 1"), "{output}");
    assert!(output.contains("Object.keys(y);"), "{output}");
    assert!(!output.contains("const o ="), "{output}");
    assert!(!output.contains("o(x)"), "{output}");
}

#[test]
fn preserves_const_alias_when_module_has_dynamic_scope() {
    // The `var` path already rejected dynamic scope; `const`/`let` aliases
    // are the same hazard: inlining reads `Object` as the global at a site
    // where `with` or a direct eval may have bound that name.
    for hazard in ["eval(code);", "with (scope) { observe(); }"] {
        let input = format!("const e = Object.freeze;\n{hazard}\nuse(e(value));\n");
        assert_eq_normalized(&apply(&input), &input);
    }
}
