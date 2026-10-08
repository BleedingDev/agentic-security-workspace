//! Static lazy-chunk URL enumeration from webpack 4/5 runtime chunk tables.
//!
//! Direct extractor tests use obviously synthetic chunk ids (101/202/303),
//! hashes (`aaaa1111` etc.), and names (`alpha`); nothing is copied from a
//! real bundle. End-to-end tests use the repo's generated webpack fixtures.

use std::fs;

use wakaru_core::unpacker::chunk_enumeration::{
    ChunkAssetKind, ChunkEnumeration, ChunkEnumerationStatus, ChunkIdSource, ChunkUrl,
    PublicPathFact, RelativeImportKind,
};
use wakaru_core::unpacker::{enumerate_chunks, extract_chunk_enumeration_from_source};
use wakaru_core::BundleFormat;

fn extract_wp5(source: &str) -> Option<ChunkEnumeration> {
    extract_chunk_enumeration_from_source(source, BundleFormat::Webpack5)
}

fn extract_wp4(source: &str) -> Option<ChunkEnumeration> {
    extract_chunk_enumeration_from_source(source, BundleFormat::Webpack4)
}

/// Minimal webpack5-shaped runtime scaffold around a `.u` assignment: the
/// candidate gate requires a function-valued binding with at least two other
/// known runtime property assignments.
fn wp5_runtime(extra: &str) -> String {
    format!(
        r#"
(() => {{
    function __webpack_require__(moduleId) {{ return moduleId; }}
    __webpack_require__.e = (chunkId) => Promise.resolve(chunkId);
    __webpack_require__.o = (obj, prop) => Object.prototype.hasOwnProperty.call(obj, prop);
    {extra}
}})();
"#
    )
}

fn url(chunk_id: &str, url: &str, source: ChunkIdSource) -> ChunkUrl {
    ChunkUrl {
        chunk_id: chunk_id.to_string(),
        url: url.to_string(),
        source,
    }
}

#[test]
fn wp5_hash_map_template_enumerates() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.p = "/assets/";
        __webpack_require__.u = (chunkId) =>
            "" + chunkId + "." + {101: "aaaa1111", 202: "bbbb2222"}[chunkId] + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::Static("/assets/".to_string())
    );
    let [asset] = enumeration.assets.as_slice() else {
        panic!("expected one asset, got {:?}", enumeration.assets);
    };
    assert_eq!(asset.kind, ChunkAssetKind::Js);
    assert_eq!(asset.status, ChunkEnumerationStatus::Enumerated);
    assert_eq!(asset.template.as_deref(), Some("[id].[map].js"));
    assert_eq!(
        asset.urls,
        vec![
            url("101", "101.aaaa1111.js", ChunkIdSource::FilenameMap),
            url("202", "202.bbbb2222.js", ChunkIdSource::FilenameMap),
        ]
    );
}

#[test]
fn wp5_name_map_with_fallback_renders_both_forms() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.u = (chunkId) =>
            ({101: "alpha"}[chunkId] || chunkId) + "." +
            {101: "aaaa1111", 202: "bbbb2222"}[chunkId] + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::NotFound);
    let asset = &enumeration.assets[0];
    assert_eq!(asset.template.as_deref(), Some("[map|id].[map].js"));
    assert_eq!(
        asset.urls,
        vec![
            url("101", "alpha.aaaa1111.js", ChunkIdSource::FilenameMap),
            url("202", "202.bbbb2222.js", ChunkIdSource::FilenameMap),
        ]
    );
}

#[test]
fn wp5_id_missing_from_strict_map_is_dropped_not_invented() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.u = (chunkId) =>
            "" + chunkId + "." + {101: "aaaa1111"}[chunkId] + ".js";
        __webpack_require__.e(303).then(__webpack_require__.bind(__webpack_require__, 9001));
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    // 303 would render `303.undefined.js` at runtime; it must be dropped.
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url("101", "101.aaaa1111.js", ChunkIdSource::FilenameMap)]
    );
}

#[test]
fn wp5_ensure_call_ids_enumerate_plain_template() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.u = (chunkId) => chunkId + ".chunk.js";
        __webpack_require__.e(101).then(__webpack_require__.bind(__webpack_require__, 9001));
        Promise.all([__webpack_require__.e(202), __webpack_require__.e("alpha")])
            .then(__webpack_require__.bind(__webpack_require__, 9002));
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    let asset = &enumeration.assets[0];
    assert_eq!(asset.status, ChunkEnumerationStatus::Enumerated);
    assert_eq!(
        asset.urls,
        vec![
            url("101", "101.chunk.js", ChunkIdSource::EnsureCall),
            url("202", "202.chunk.js", ChunkIdSource::EnsureCall),
            url("alpha", "alpha.chunk.js", ChunkIdSource::EnsureCall),
        ]
    );
}

#[test]
fn wp5_unrelated_dot_e_then_call_is_not_a_chunk_id_source() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.u = (chunkId) => chunkId + ".chunk.js";
        var client = globalThis.authClient;
        client.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    let asset = &enumeration.assets[0];
    assert_eq!(asset.status, ChunkEnumerationStatus::NoStaticChunkIds);
    assert!(asset.urls.is_empty());
}

#[test]
fn wp5_ensure_receiver_must_match_bound_continuation() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.u = (chunkId) => chunkId + ".chunk.js";
        var client = globalThis.authClient;
        client.e(101).then(__webpack_require__.bind(__webpack_require__, 9001));
        Promise.all([__webpack_require__.e(202), client.e(303)])
            .then(__webpack_require__.bind(__webpack_require__, 9002));
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    let asset = &enumeration.assets[0];
    assert_eq!(asset.status, ChunkEnumerationStatus::NoStaticChunkIds);
    assert!(asset.urls.is_empty());
}

#[test]
fn wp5_literal_template_without_id_source_fails_closed() {
    let source = wp5_runtime(r#"__webpack_require__.u = (chunkId) => chunkId + ".js";"#);
    let enumeration = extract_wp5(&source).expect("enumeration");
    let asset = &enumeration.assets[0];
    assert_eq!(asset.status, ChunkEnumerationStatus::NoStaticChunkIds);
    assert_eq!(asset.template.as_deref(), Some("[id].js"));
    assert!(asset.urls.is_empty());
}

#[test]
fn wp5_dynamic_template_fails_closed() {
    let source = wp5_runtime(
        r#"
        var suffix = globalThis.someSuffix;
        __webpack_require__.u = (chunkId) => chunkId + suffix;
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    let asset = &enumeration.assets[0];
    assert_eq!(asset.status, ChunkEnumerationStatus::DynamicTemplate);
    assert_eq!(asset.template, None);
    assert!(asset.urls.is_empty());
}

#[test]
fn wp5_runtime_computed_public_path_is_not_guessed() {
    let source = wp5_runtime(
        r#"
        var scriptUrl = globalThis.location + "";
        __webpack_require__.p = scriptUrl;
        __webpack_require__.u = (chunkId) => "" + chunkId + "." + {101: "aaaa1111"}[chunkId] + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_script_relative_public_path_extracts_literal_suffix() {
    // webpack's automatic publicPath is the script's own directory plus a
    // configured relative suffix (here `../../`, climbing to the site root).
    let source = wp5_runtime(
        r#"
        var scriptUrl = document.currentScript.src;
        scriptUrl = scriptUrl
            .replace(/^blob:/, "")
            .replace(/#.*$/, "")
            .replace(/\?.*$/, "")
            .replace(/\/[^\/]+$/, "/");
        __webpack_require__.p = scriptUrl + "../../";
        __webpack_require__.u = (chunkId) =>
            "static/js/" + chunkId + "." + {101: "aaaa1111"}[chunkId] + ".chunk.js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::ScriptRelative("../../".to_string())
    );
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url(
            "101",
            "static/js/101.aaaa1111.chunk.js",
            ChunkIdSource::FilenameMap
        )]
    );
}

#[test]
fn wp5_script_relative_traces_through_reassignment_chain() {
    // The minified form of webpack's automatic publicPath assigns the script
    // URL and then trims it through reassignments before the `.p` write.
    let source = wp5_runtime(
        r#"
        var t;
        t = document.currentScript.src;
        t = t
            .replace(/^blob:/, "")
            .replace(/#.*$/, "")
            .replace(/\?.*$/, "")
            .replace(/\/[^\/]+$/, "/");
        __webpack_require__.p = t + "../../";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::ScriptRelative("../../".to_string())
    );
}

#[test]
fn wp5_script_relative_accepts_runtime_document_script_fallback() {
    let source = wp5_runtime(
        r#"
        var runtimeDocument = __webpack_require__.g.document;
        var scripts = runtimeDocument.getElementsByTagName("script");
        var scriptUrl = scripts[0].src;
        scriptUrl = scriptUrl
            .replace(/^blob:/, "")
            .replace(/#.*$/, "")
            .replace(/\?.*$/, "")
            .replace(/\/[^\/]+$/, "/");
        __webpack_require__.p = scriptUrl + "../../";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::ScriptRelative("../../".to_string())
    );
}

#[test]
fn wp5_script_relative_accepts_minified_inline_public_path_assignment() {
    let source = wp5_runtime(
        r#"
        var scriptUrl;
        __webpack_require__.g.importScripts &&
            (scriptUrl = __webpack_require__.g.location + "");
        var runtimeDocument = __webpack_require__.g.document;
        if (!scriptUrl && runtimeDocument && runtimeDocument.currentScript) {
            scriptUrl = runtimeDocument.currentScript.src;
        }
        __webpack_require__.p = (scriptUrl = scriptUrl
            .replace(/^blob:/, "")
            .replace(/#.*$/, "")
            .replace(/\?.*$/, "")
            .replace(/\/[^\/]+$/, "/")) + "../../";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::ScriptRelative("../../".to_string())
    );
}

#[test]
fn wp5_untrimmed_script_url_is_not_a_script_directory() {
    let source = wp5_runtime(
        r#"
        var scriptUrl = document.currentScript.src;
        __webpack_require__.p = scriptUrl + "assets/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_arbitrary_src_member_is_not_a_script_url() {
    let source = wp5_runtime(
        r#"
        var base = globalThis.appConfig.src;
        base = base.replace(/\/[^\/]+$/, "/");
        __webpack_require__.p = base + "assets/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_same_name_in_another_scope_does_not_vouch_for_public_path_base() {
    let source = wp5_runtime(
        r#"
        function unrelated() {
            var base = document.currentScript.src;
            base = base.replace(/\/[^\/]+$/, "/");
            return base;
        }
        var base = globalThis.cdnBase;
        __webpack_require__.p = base + "assets/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_literal_concat_public_path_is_not_script_relative() {
    // A fully-literal concatenation is a static value, not script-relative.
    let source = wp5_runtime(
        r#"
        __webpack_require__.p = "/assets/" + "js/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    // The left operand is a string literal, so this is not script-relative;
    // it falls through to RuntimeComputed (multi-part static is not folded).
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_chained_concat_public_path_is_not_script_relative() {
    // `a + b + "c"` parses left-associatively; claiming only `"c"` would
    // silently drop the middle runtime operand. Fail closed instead.
    let source = wp5_runtime(
        r#"
        var scriptUrl = globalThis.location + "";
        var basePath = computeBasePath();
        __webpack_require__.p = scriptUrl + basePath + "/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_non_script_url_base_is_not_script_relative() {
    // An app-configured `__webpack_public_path__` compiles to the same
    // `<expr> + "<literal>"` shape, but its base is not the script URL —
    // reading the suffix as script-relative would fabricate a path.
    let source = wp5_runtime(
        r#"
        __webpack_require__.p = window.__CDN_BASE__ + "static/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_untraced_ident_base_is_not_script_relative() {
    // Same shape with a bare identifier base: unless that identifier
    // provably received a script-URL-derived value, fail closed.
    let source = wp5_runtime(
        r#"
        var cdnBase = window.__CDN_BASE__;
        __webpack_require__.p = cdnBase + "static/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_absolute_suffix_is_not_script_relative() {
    // A suffix starting with `/` resolves against the origin root, not the
    // script directory — the script-relative replay identity breaks.
    let source = wp5_runtime(
        r#"
        var scriptUrl = globalThis.location + "";
        __webpack_require__.p = scriptUrl + "/assets/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_condition_only_mention_is_not_traced() {
    // A script-URL source appearing only in a ternary condition does not
    // flow into the assigned value.
    let source = wp5_runtime(
        r#"
        var b = document.currentScript ? window.__CDN_BASE__ : "cdn/";
        __webpack_require__.p = b + "static/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_nested_function_mention_is_not_traced() {
    // A script-URL source inside a nested function body is not the assigned
    // value; the identifier holds a function.
    let source = wp5_runtime(
        r#"
        var t = function () { return document.currentScript.src; };
        __webpack_require__.p = t + "static/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_method_body_mention_is_not_traced() {
    // Same for class and object-literal method bodies: the identifier holds
    // a class or object, not a script URL.
    let source = wp5_runtime(
        r#"
        var t = class { m() { return document.currentScript.src; } };
        var u = { m() { return document.currentScript.src; } };
        __webpack_require__.p = t + "static/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_logical_and_guard_mention_is_not_traced() {
    // `a && b` yields `b`; a script-URL source in the guard position does
    // not flow into the value.
    let source = wp5_runtime(
        r#"
        var t = someEl.src && window.__CDN_BASE__;
        __webpack_require__.p = t + "static/";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_non_directory_suffix_is_not_script_relative() {
    // webpack's automatic publicPath is always a directory path ending in
    // `/`; a bare filename-ish or query suffix is not that shape.
    let source = wp5_runtime(
        r#"
        var scriptUrl = globalThis.location + "";
        __webpack_require__.p = scriptUrl + "assets";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_empty_suffix_is_not_script_relative() {
    // `scriptUrl + ""` carries no provable suffix; an empty claim is
    // semantically RuntimeComputed and should be reported as such.
    let source = wp5_runtime(
        r#"
        var scriptUrl = globalThis.location + "";
        __webpack_require__.p = scriptUrl + "";
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.e(101).then(() => 0);
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
}

#[test]
fn wp5_mini_css_ids_come_only_from_its_own_map() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.u = (chunkId) => chunkId + ".js";
        __webpack_require__.miniCssF = (chunkId) =>
            "css/" + chunkId + "." + {101: "cccc3333"}[chunkId] + ".css";
        __webpack_require__.e(202).then(__webpack_require__.bind(__webpack_require__, 9001));
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    let [js, css] = enumeration.assets.as_slice() else {
        panic!("expected js + css assets, got {:?}", enumeration.assets);
    };
    assert_eq!(js.kind, ChunkAssetKind::Js);
    assert_eq!(
        js.urls,
        vec![url("202", "202.js", ChunkIdSource::EnsureCall)]
    );
    assert_eq!(css.kind, ChunkAssetKind::Css);
    // The `.e(202)` load site proves a JS request, not a CSS asset.
    assert_eq!(
        css.urls,
        vec![url(
            "101",
            "css/101.cccc3333.css",
            ChunkIdSource::FilenameMap
        )]
    );
}

#[test]
fn wp5_template_literal_form_is_supported() {
    let source = wp5_runtime(
        r#"
        __webpack_require__.u = (chunkId) => `${chunkId}.tpl.js`;
        __webpack_require__.e(101).then(__webpack_require__.bind(__webpack_require__, 9001));
        "#,
    );
    let enumeration = extract_wp5(&source).expect("enumeration");
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url("101", "101.tpl.js", ChunkIdSource::EnsureCall)]
    );
}

#[test]
fn wp5_multiple_runtime_candidates_fail_closed() {
    let source = r#"
(() => {
    function alphaRequire(id) { return id; }
    alphaRequire.e = (c) => Promise.resolve(c);
    alphaRequire.o = (o, p) => true;
    alphaRequire.u = (c) => c + ".a.js";
    function betaRequire(id) { return id; }
    betaRequire.e = (c) => Promise.resolve(c);
    betaRequire.o = (o, p) => true;
    betaRequire.u = (c) => c + ".b.js";
})();
"#;
    assert_eq!(extract_wp5(source), None);
}

#[test]
fn wp5_bare_u_assignment_without_runtime_surface_fails_closed() {
    let source = r#"
(() => {
    function notWebpack(id) { return id; }
    notWebpack.u = (c) => c + ".js";
})();
"#;
    assert_eq!(extract_wp5(source), None);
}

#[test]
fn non_webpack_formats_are_not_enumerated() {
    let source = wp5_runtime(r#"__webpack_require__.u = (chunkId) => chunkId + ".js";"#);
    assert_eq!(
        extract_chunk_enumeration_from_source(&source, BundleFormat::Browserify),
        None
    );
}

#[test]
fn wp4_jsonp_script_src_enumerates_with_maps() {
    let source = r#"
(function () {
    function jsonpScriptSrc(chunkId) {
        return __webpack_require__.p + "" +
            ({101: "alpha"}[chunkId] || chunkId) + "." +
            {101: "aaaa1111", 202: "bbbb2222"}[chunkId] + ".chunk.js";
    }
    function __webpack_require__(moduleId) { return moduleId; }
    __webpack_require__.p = "/static/";
    __webpack_require__.e(202).then(__webpack_require__.t.bind(null, 9001, 7));
})();
"#;
    let enumeration = extract_wp4(source).expect("enumeration");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::Static("/static/".to_string())
    );
    let [asset] = enumeration.assets.as_slice() else {
        panic!("expected one asset, got {:?}", enumeration.assets);
    };
    assert_eq!(asset.template.as_deref(), Some("[map|id].[map].chunk.js"));
    assert_eq!(
        asset.urls,
        vec![
            url("101", "alpha.aaaa1111.chunk.js", ChunkIdSource::FilenameMap),
            url("202", "202.bbbb2222.chunk.js", ChunkIdSource::FilenameMap),
        ]
    );
}

#[test]
fn wp4_inlined_script_src_expression_is_recognized() {
    // Terser inlines `jsonpScriptSrc` into an immediately invoked function
    // expression at the `script.src` assignment.
    let source = r#"
(function () {
    function __webpack_require__(moduleId) { return moduleId; }
    __webpack_require__.p = "";
    var script = {};
    script.src = function (chunkId) {
        return __webpack_require__.p + "" + chunkId + ".bundle.js";
    }(101);
    __webpack_require__.e(101).then(__webpack_require__.bind(null, 9001));
})();
"#;
    let enumeration = extract_wp4(source).expect("enumeration");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::Static(String::new())
    );
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url("101", "101.bundle.js", ChunkIdSource::EnsureCall)]
    );
}

#[test]
fn wp4_conflicting_script_src_candidates_fail_closed() {
    let source = r#"
(function () {
    function __webpack_require__(moduleId) { return moduleId; }
    __webpack_require__.p = "";
    function alphaSrc(chunkId) { return __webpack_require__.p + chunkId + ".a.js"; }
    function betaSrc(chunkId) { return __webpack_require__.p + chunkId + ".b.js"; }
    alphaSrc(101); betaSrc(101);
})();
"#;
    assert_eq!(extract_wp4(source), None);
}

// ---------------------------------------------------------------------------
// End-to-end: detection-gated enumeration through the dedicated debug path
// ---------------------------------------------------------------------------

fn enumerate_fixture(path: &str) -> Option<ChunkEnumeration> {
    let full = format!("tests/bundles/webpack-gen/dist/{path}");
    let source = fs::read_to_string(&full).unwrap_or_else(|e| panic!("failed to read {full}: {e}"));
    enumerate_chunks(&source, path)
        .unwrap_or_else(|e| panic!("enumeration should succeed for {path}: {e}"))
        .enumeration
}

fn fixture_enumeration(path: &str) -> ChunkEnumeration {
    enumerate_fixture(path).unwrap_or_else(|| panic!("expected one chunk enumeration for {path}"))
}

#[test]
fn wp5_dynamic_fixture_enumerates_named_chunk() {
    let enumeration = fixture_enumeration("wp5-dynamic/bundle.js");
    assert_eq!(enumeration.public_path, PublicPathFact::NotFound);
    let js = &enumeration.assets[0];
    assert_eq!(js.status, ChunkEnumerationStatus::Enumerated);
    assert_eq!(
        js.urls,
        vec![url(
            "src_greet_js",
            "src_greet_js.bundle.js",
            ChunkIdSource::EnsureCall
        )]
    );
}

#[test]
fn wp5_array_fixture_reports_runtime_public_path() {
    let enumeration = fixture_enumeration("wp5-array/bundle.js");
    assert_eq!(enumeration.public_path, PublicPathFact::RuntimeComputed);
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url("1", "chunk-1.js", ChunkIdSource::EnsureCall)]
    );
}

#[test]
fn wp5_dynamic_min_fixture_enumerates() {
    let enumeration = fixture_enumeration("wp5-dynamic-min/bundle.js");
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url("529", "529.bundle.js", ChunkIdSource::EnsureCall)]
    );
}

#[test]
fn wp4_dynamic_fixture_enumerates() {
    let enumeration = fixture_enumeration("wp4-dynamic/bundle.js");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::Static(String::new())
    );
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url("0", "0.bundle.js", ChunkIdSource::EnsureCall)]
    );
}

#[test]
fn wp4_dynamic_min_fixture_enumerates() {
    let enumeration = fixture_enumeration("wp4-dynamic-min/bundle.js");
    assert_eq!(
        enumeration.public_path,
        PublicPathFact::Static(String::new())
    );
    assert_eq!(
        enumeration.assets[0].urls,
        vec![url("1", "1.bundle.js", ChunkIdSource::EnsureCall)]
    );
}

#[test]
fn non_chunked_fixture_has_no_enumeration() {
    let enumeration = enumerate_fixture("wp5-cjs/bundle.js");
    assert!(
        enumeration.is_none(),
        "no chunk-filename runtime expected: {enumeration:?}"
    );
}

// ---------------------------------------------------------------------------
// Native-ESM (Vite/Rollup/rolldown/esbuild) literal relative specifiers
// ---------------------------------------------------------------------------

fn enumerate_source(source: &str) -> Option<ChunkEnumeration> {
    enumerate_chunks(source, "entry.js")
        .expect("enumeration should succeed")
        .enumeration
}

fn source_enumeration(source: &str) -> ChunkEnumeration {
    enumerate_source(source).expect("expected one enumeration")
}

#[test]
fn native_esm_static_and_dynamic_relative_imports_enumerate() {
    // Rollup/Vite-shaped entry: static sibling imports plus a lazy import().
    let source = r#"
import { a as x } from "./aaaa1111.js";
export { y } from "./bbbb2222.js";
export * from "./cccc3333.js";
const load = () => import("./lazy-beta.js");
console.log(x, load);
"#;
    let enumeration = source_enumeration(source);
    assert_eq!(enumeration.public_path, PublicPathFact::NotFound);
    assert!(enumeration.assets.is_empty());
    let seen: Vec<(&str, RelativeImportKind)> = enumeration
        .relative_imports
        .iter()
        .map(|import| (import.specifier.as_str(), import.kind))
        .collect();
    assert_eq!(
        seen,
        vec![
            ("./aaaa1111.js", RelativeImportKind::Import),
            ("./bbbb2222.js", RelativeImportKind::ExportFrom),
            ("./cccc3333.js", RelativeImportKind::ExportFrom),
            ("./lazy-beta.js", RelativeImportKind::DynamicImport),
        ]
    );
}

#[test]
fn bare_and_absolute_specifiers_are_not_enumerated() {
    // Only relative specifiers are sibling chunks; packages and absolute URLs
    // are not fetchable siblings and must be excluded.
    let source = r#"
import react from "react";
import x from "https://cdn.example.com/lib.js";
import y from "/absolute/app.js";
const z = import("some-package");
console.log(react, x, y, z);
"#;
    let enumeration = enumerate_source(source);
    assert!(
        enumeration.is_none(),
        "no relative specifiers expected: {enumeration:?}"
    );
}

#[test]
fn dynamic_import_of_expression_fails_closed() {
    let source = r#"
const name = globalThis.chunkName;
const a = import(name);
const b = import("./" + name);
export { a, b };
"#;
    let enumeration = enumerate_source(source);
    assert!(
        enumeration.is_none(),
        "computed import specifiers must not be enumerated: {enumeration:?}"
    );
}

#[test]
fn duplicate_relative_specifier_kinds_are_preserved_once_each() {
    let source = r#"
import { a } from "./shared-gamma.js";
import { b } from "./shared-gamma.js";
const c = () => import("./shared-gamma.js");
export { a, b, c };
"#;
    let enumeration = source_enumeration(source);
    assert_eq!(enumeration.relative_imports.len(), 2);
    assert_eq!(
        enumeration.relative_imports[0].specifier,
        "./shared-gamma.js"
    );
    assert_eq!(
        enumeration.relative_imports[0].kind,
        RelativeImportKind::Import
    );
    assert_eq!(
        enumeration.relative_imports[1].specifier,
        "./shared-gamma.js"
    );
    assert_eq!(
        enumeration.relative_imports[1].kind,
        RelativeImportKind::DynamicImport
    );
}

#[test]
fn parent_relative_specifiers_are_enumerated() {
    let source = r#"import { a } from "../vendor/ffff6666.js"; console.log(a);"#;
    let enumeration = source_enumeration(source);
    assert_eq!(
        enumeration.relative_imports[0].specifier,
        "../vendor/ffff6666.js"
    );
}
