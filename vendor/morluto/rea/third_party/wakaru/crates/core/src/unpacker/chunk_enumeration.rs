//! Static enumeration of chunk references from webpack runtime tables and
//! literal relative ESM specifiers.
//!
//! Webpack emits the chunk filename template as a runtime function —
//! `__webpack_require__.u` / `.miniCssF` (webpack 5) or `jsonpScriptSrc`
//! (webpack 4) — whose body is a concatenation of string literals, the
//! `chunkId` parameter, and object-literal lookups keyed by chunk id. When
//! that template and its id sources are fully literal, the exact relative
//! URLs the runtime would request can be enumerated without executing any
//! code.
//!
//! Everything here is fail-closed: a template with any non-literal part is
//! reported as [`ChunkEnumerationStatus::DynamicTemplate`] with no URLs, a
//! literal template with no static chunk-id source is reported as
//! [`ChunkEnumerationStatus::NoStaticChunkIds`], and a runtime-computed
//! `publicPath` is reported as [`PublicPathFact::RuntimeComputed`]; in that
//! state the filename suffix is known but the complete request URL is not.
//! The one recognized runtime shape is webpack's automatic publicPath — an
//! identifier binding traced from a script-URL source through webpack's
//! canonical filename-to-directory normalization plus a literal relative
//! suffix — reported as [`PublicPathFact::ScriptRelative`] carrying only the
//! suffix.
//! No URL is ever guessed or synthesized beyond what the template proves.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use swc_core::common::Mark;
use swc_core::ecma::ast::{
    ArrowFunctionBody, AssignExpr, AssignOp, AssignTarget, BinaryOp, CallExpr, Callee, Expr,
    FnDecl, Function, Id, Ident, Lit, MemberExpr, MemberProp, MetaPropKind, Module, ModuleDecl,
    ModuleItem, ObjectLit, Pat, Prop, PropName, PropOrSpread, SimpleAssignTarget, Stmt,
    VarDeclarator,
};
use swc_core::ecma::visit::{Visit, VisitWith};

use crate::unpacker::webpack5::is_webpack_runtime_property;
use crate::unpacker::BundleFormat;
use crate::utils::paren::strip_parens;

/// Statically extracted chunk-reference surface of one input.
///
/// Carries two independent, both fail-closed, sources:
/// - `assets` — webpack 4/5 runtime chunk-filename table enumeration.
/// - `relative_imports` — literal relative ESM module specifiers (native
///   code-splitting, e.g. Vite/Rollup/rolldown/esbuild output), where the
///   specifier *is* the chunk URL.
///
/// A `None` enumeration (absence) means neither source produced anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkEnumeration {
    pub public_path: PublicPathFact,
    pub assets: Vec<ChunkAssetEnumeration>,
    pub relative_imports: Vec<RelativeImport>,
}

/// Merge a detected webpack runtime surface with format-independent literal
/// relative ESM references. Absence means neither source produced a fact.
pub(super) fn merge(
    webpack: Option<ChunkEnumeration>,
    relative_imports: Vec<RelativeImport>,
) -> Option<ChunkEnumeration> {
    match webpack {
        Some(mut enumeration) => {
            enumeration.relative_imports = relative_imports;
            Some(enumeration)
        }
        None if !relative_imports.is_empty() => Some(ChunkEnumeration {
            public_path: PublicPathFact::NotFound,
            assets: Vec::new(),
            relative_imports,
        }),
        None => None,
    }
}

/// One literal relative module specifier imported by the bundle. For native
/// ESM code-splitting the specifier is the exact relative URL of a sibling
/// chunk; `public_path` does not apply — resolve it against the entry URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativeImport {
    /// The specifier verbatim (e.g. `./bbbb2222.js`).
    pub specifier: String,
    pub kind: RelativeImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelativeImportKind {
    /// `import ... from "./x"`.
    Import,
    /// `export ... from "./x"` / `export * from "./x"`.
    ExportFrom,
    /// `import("./x")` — a dynamic import (a genuine lazy chunk).
    DynamicImport,
}

/// What could be proven about `__webpack_require__.p`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicPathFact {
    /// Exactly one assignment, with a string-literal value.
    Static(String),
    /// A single `<script-directory binding> + "<literal suffix>"` assignment.
    /// The binding is scope-resolved, originates at webpack's generated
    /// script-URL sources, and passes through its canonical
    /// `replace(/\/[^\/]+$/, "/")` directory normalization. The suffix is
    /// non-empty, relative, and `/`-terminated. Consumers may resolve
    /// `suffix + chunkUrl` against the entry URL.
    ScriptRelative(String),
    /// Assigned, but not provable to a single literal or canonical
    /// script-relative value. The filename template remains useful, but the
    /// complete request URL cannot be resolved statically.
    RuntimeComputed,
    /// No assignment found.
    NotFound,
}

/// One asset-kind enumeration (JS chunks, or mini-css-extract CSS chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkAssetEnumeration {
    pub kind: ChunkAssetKind,
    pub status: ChunkEnumerationStatus,
    /// Debug-oriented placeholder rendering of the filename template
    /// (`[id]` for the chunk id, `[map]` for a static lookup table,
    /// `[map|id]` for a lookup with an id fallback). `None` when the
    /// template is dynamic. Not machine-parseable; `urls` is the contract.
    pub template: Option<String>,
    /// Relative URLs exactly as the runtime template renders them.
    /// `public_path` is deliberately not prepended.
    pub urls: Vec<ChunkUrl>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkAssetKind {
    Js,
    Css,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkEnumerationStatus {
    /// Literal template with at least one static chunk-id source.
    Enumerated,
    /// Literal template, but no static lookup table and no literal
    /// `.e(id)` load sites — the id universe is unknowable statically.
    NoStaticChunkIds,
    /// The filename function is not a pure literal template.
    DynamicTemplate,
}

/// One enumerated chunk URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkUrl {
    pub chunk_id: String,
    pub url: String,
    pub source: ChunkIdSource,
}

/// Where a chunk id was proven from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkIdSource {
    /// A key of a lookup table inside the filename template itself.
    FilenameMap,
    /// A literal argument of a generated, binding-correlated
    /// `X.e(id).then(X.bind(...))` async chunk load.
    EnsureCall,
}

/// Extract the chunk enumeration for one detected bundle input.
///
/// Only webpack 4/5 runtimes are understood; other formats return `None`.
pub fn extract_chunk_enumeration(
    module: &Module,
    format: BundleFormat,
    unresolved_mark: Mark,
) -> Option<ChunkEnumeration> {
    match format {
        BundleFormat::Webpack5 => extract_webpack5(module, unresolved_mark),
        BundleFormat::Webpack4 => extract_webpack4(module, unresolved_mark),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Template model
// ---------------------------------------------------------------------------

/// One segment of a literal chunk-filename template.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    Lit(String),
    ChunkId,
    Map {
        entries: BTreeMap<String, String>,
        /// `{...}[chunkId] || chunkId` — ids absent from the table render as
        /// themselves instead of excluding the id.
        fallback: bool,
    },
}

/// Parse a filename function value (`(chunkId) => expr` or
/// `function (chunkId) { return expr; }`) into literal segments.
/// `None` means the template is not statically evaluable.
fn parse_filename_fn(expr: &Expr) -> Option<Vec<Segment>> {
    let (param, body) = filename_fn_parts(expr)?;
    let mut segments = Vec::new();
    parse_segments(body, &param, &mut segments).then_some(segments)
}

/// Split a candidate filename function into its single parameter and the
/// single returned expression.
fn filename_fn_parts(expr: &Expr) -> Option<(Id, &Expr)> {
    match strip_parens(expr) {
        Expr::Arrow(arrow) => arrow_parts(arrow),
        Expr::Fn(fn_expr) => function_parts(&fn_expr.function),
        _ => None,
    }
}

/// Split a one-parameter arrow with a single-expression body into its
/// parameter id and returned expression.
fn arrow_parts(arrow: &swc_core::ecma::ast::ArrowExpr) -> Option<(Id, &Expr)> {
    let [Pat::Ident(param)] = arrow.params.as_slice() else {
        return None;
    };
    let body = match &*arrow.body {
        ArrowFunctionBody::Expr(expr) => strip_parens(expr),
        ArrowFunctionBody::FunctionBody(block) => single_return_expr(&block.stmts)?,
    };
    Some((param.id.to_id(), body))
}

/// Split a one-parameter function with a single-return body into its
/// parameter id and returned expression.
fn function_parts(function: &Function) -> Option<(Id, &Expr)> {
    let [param] = function.params.as_slice() else {
        return None;
    };
    let Pat::Ident(param) = &param.pat else {
        return None;
    };
    let body = single_return_expr(&function.body.as_ref()?.stmts)?;
    Some((param.id.to_id(), body))
}

fn single_return_expr(stmts: &[Stmt]) -> Option<&Expr> {
    let [Stmt::Return(ret)] = stmts else {
        return None;
    };
    Some(strip_parens(ret.arg.as_deref()?))
}

/// Recursively fold a concatenation expression into segments. Returns `false`
/// (dynamic) on the first construct that is not a string literal, the chunk-id
/// parameter, or a static object-literal lookup keyed by that parameter.
fn parse_segments(expr: &Expr, param: &Id, out: &mut Vec<Segment>) -> bool {
    let expr = strip_parens(expr);
    match expr {
        Expr::Bin(bin) if bin.op == BinaryOp::Add => {
            parse_segments(&bin.left, param, out) && parse_segments(&bin.right, param, out)
        }
        Expr::Lit(Lit::Str(value)) => {
            let Some(text) = value.value.as_str() else {
                return false;
            };
            out.push(Segment::Lit(text.to_string()));
            true
        }
        Expr::Ident(ident) if ident.to_id() == *param => {
            out.push(Segment::ChunkId);
            true
        }
        Expr::Tpl(tpl) => {
            // Interleave quasis and expressions in source order.
            for (index, quasi) in tpl.quasis.iter().enumerate() {
                let Some(cooked) = quasi.cooked.as_ref().and_then(|cooked| cooked.as_str()) else {
                    return false;
                };
                out.push(Segment::Lit(cooked.to_string()));
                if let Some(inner) = tpl.exprs.get(index) {
                    if !parse_segments(inner, param, out) {
                        return false;
                    }
                }
            }
            true
        }
        Expr::Bin(bin) if bin.op == BinaryOp::LogicalOr => {
            let Some(entries) = map_lookup_entries(&bin.left, param) else {
                return false;
            };
            let fallback = matches!(
                strip_parens(&bin.right), Expr::Ident(ident) if ident.to_id() == *param
            );
            if !fallback {
                return false;
            }
            out.push(Segment::Map {
                entries,
                fallback: true,
            });
            true
        }
        Expr::Member(_) => {
            let Some(entries) = map_lookup_entries(expr, param) else {
                return false;
            };
            out.push(Segment::Map {
                entries,
                fallback: false,
            });
            true
        }
        _ => false,
    }
}

/// Match `{...literal entries...}[chunkId]` and return its entries.
fn map_lookup_entries(expr: &Expr, param: &Id) -> Option<BTreeMap<String, String>> {
    let Expr::Member(MemberExpr { obj, prop, .. }) = strip_parens(expr) else {
        return None;
    };
    let Expr::Object(object) = strip_parens(obj) else {
        return None;
    };
    let MemberProp::Computed(computed) = prop else {
        return None;
    };
    let Expr::Ident(key) = strip_parens(&computed.expr) else {
        return None;
    };
    if key.to_id() != *param {
        return None;
    }
    object_literal_entries(object)
}

/// Read an object literal as `chunk id -> literal string` entries.
/// Any non-literal key or value fails the whole table closed.
fn object_literal_entries(object: &ObjectLit) -> Option<BTreeMap<String, String>> {
    let mut entries = BTreeMap::new();
    for prop in &object.props {
        let PropOrSpread::Prop(prop) = prop else {
            return None;
        };
        let Prop::KeyValue(key_value) = &**prop else {
            return None;
        };
        let key = match &key_value.key {
            PropName::Ident(ident) => ident.sym.as_str().to_string(),
            PropName::Str(value) => value.value.as_str()?.to_string(),
            PropName::Num(value) => integer_string(value.value)?,
            _ => return None,
        };
        let value = match strip_parens(&key_value.value) {
            Expr::Lit(Lit::Str(value)) => value.value.as_str()?.to_string(),
            Expr::Lit(Lit::Num(value)) => integer_string(value.value)?,
            _ => return None,
        };
        entries.insert(key, value);
    }
    Some(entries)
}

/// Render an f64 as the canonical JS string of a non-negative integer.
/// Non-integers fail closed rather than risking a formatting mismatch.
fn integer_string(value: f64) -> Option<String> {
    (value.fract() == 0.0 && (0.0..=9007199254740992.0).contains(&value))
        .then(|| format!("{}", value as u64))
}

fn render_template(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|segment| match segment {
            Segment::Lit(value) => value.as_str(),
            Segment::ChunkId => "[id]",
            Segment::Map {
                fallback: false, ..
            } => "[map]",
            Segment::Map { fallback: true, .. } => "[map|id]",
        })
        .collect()
}

/// Expand a parsed template over its provable chunk-id universe.
///
/// `extra_ids` carries literal `.e(id)` load-site ids (JS chunks only). An id
/// missing from a strict (no-fallback) lookup table is dropped entirely — the
/// runtime would render `"undefined"` there, which is not a real URL.
fn enumerate_asset(
    kind: ChunkAssetKind,
    segments: Vec<Segment>,
    extra_ids: &BTreeSet<String>,
) -> ChunkAssetEnumeration {
    let mut map_ids: BTreeSet<String> = BTreeSet::new();
    for segment in &segments {
        if let Segment::Map { entries, .. } = segment {
            map_ids.extend(entries.keys().cloned());
        }
    }
    let mut all_ids = map_ids.clone();
    all_ids.extend(extra_ids.iter().cloned());
    if all_ids.is_empty() {
        return ChunkAssetEnumeration {
            kind,
            status: ChunkEnumerationStatus::NoStaticChunkIds,
            template: Some(render_template(&segments)),
            urls: Vec::new(),
        };
    }

    let mut urls = Vec::new();
    'ids: for id in &all_ids {
        let mut url = String::new();
        for segment in &segments {
            match segment {
                Segment::Lit(value) => url.push_str(value),
                Segment::ChunkId => url.push_str(id),
                Segment::Map { entries, fallback } => match entries.get(id) {
                    Some(value) => url.push_str(value),
                    None if *fallback => url.push_str(id),
                    None => continue 'ids,
                },
            }
        }
        urls.push(ChunkUrl {
            chunk_id: id.clone(),
            url,
            source: if map_ids.contains(id) {
                ChunkIdSource::FilenameMap
            } else {
                ChunkIdSource::EnsureCall
            },
        });
    }
    ChunkAssetEnumeration {
        kind,
        status: ChunkEnumerationStatus::Enumerated,
        template: Some(render_template(&segments)),
        urls,
    }
}

fn dynamic_asset(kind: ChunkAssetKind) -> ChunkAssetEnumeration {
    ChunkAssetEnumeration {
        kind,
        status: ChunkEnumerationStatus::DynamicTemplate,
        template: None,
        urls: Vec::new(),
    }
}

fn valid_script_relative_suffix(suffix: &str) -> bool {
    !suffix.is_empty()
        && suffix.ends_with('/')
        && !suffix.starts_with('/')
        && !suffix.contains("://")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptTraceValue {
    Document,
    ScriptCollection,
    ScriptElement,
    ScriptUrl,
    ScriptDirectory,
}

/// Binding-aware recognition of webpack's generated automatic-publicPath
/// prelude. A script-relative claim requires both an exact script-URL source
/// and webpack's canonical filename-to-directory replacement; merely seeing a
/// `.src` property or an untrimmed `document.currentScript.src` is not enough.
struct ScriptPathTrace {
    runtime: Id,
    unresolved_mark: Mark,
    values: HashMap<Id, ScriptTraceValue>,
    public_path_assignments: Vec<PublicPathFact>,
    allow_runtime_location: bool,
}

impl ScriptPathTrace {
    fn collect(module: &Module, runtime: Id, unresolved_mark: Mark) -> PublicPathFact {
        let mut trace = Self {
            runtime,
            unresolved_mark,
            values: HashMap::new(),
            public_path_assignments: Vec::new(),
            allow_runtime_location: false,
        };
        module.visit_with(&mut trace);
        match trace.public_path_assignments.as_slice() {
            [] => PublicPathFact::NotFound,
            [single] => single.clone(),
            _ => PublicPathFact::RuntimeComputed,
        }
    }

    fn set_binding(&mut self, id: Id, value: Option<ScriptTraceValue>) {
        match value {
            Some(value) => {
                self.values.insert(id, value);
            }
            None => {
                self.values.remove(&id);
            }
        }
    }

    fn classify_public_path(&self, expr: &Expr) -> PublicPathFact {
        if let Expr::Lit(Lit::Str(value)) = strip_parens(expr) {
            return value
                .value
                .as_str()
                .map(|value| PublicPathFact::Static(value.to_string()))
                .unwrap_or(PublicPathFact::RuntimeComputed);
        }
        let Expr::Bin(bin) = strip_parens(expr) else {
            return PublicPathFact::RuntimeComputed;
        };
        if bin.op != BinaryOp::Add {
            return PublicPathFact::RuntimeComputed;
        }
        let Expr::Lit(Lit::Str(suffix)) = strip_parens(&bin.right) else {
            return PublicPathFact::RuntimeComputed;
        };
        let Some(suffix) = suffix.value.as_str() else {
            return PublicPathFact::RuntimeComputed;
        };
        if self.classify_value(&bin.left) == Some(ScriptTraceValue::ScriptDirectory)
            && valid_script_relative_suffix(suffix)
        {
            PublicPathFact::ScriptRelative(suffix.to_string())
        } else {
            PublicPathFact::RuntimeComputed
        }
    }

    fn classify_value(&self, expr: &Expr) -> Option<ScriptTraceValue> {
        let expr = strip_parens(expr);
        match expr {
            Expr::Ident(ident) => self.values.get(&ident.to_id()).copied().or_else(|| {
                self.is_global_ident(ident, "document")
                    .then_some(ScriptTraceValue::Document)
            }),
            Expr::Member(member) => self.classify_member(member),
            Expr::Call(call) => self.classify_call(call),
            Expr::Bin(bin) if bin.op == BinaryOp::Add => {
                if is_empty_string(&bin.left) {
                    self.classify_value(&bin.right)
                } else if is_empty_string(&bin.right) {
                    self.classify_value(&bin.left)
                } else {
                    None
                }
            }
            Expr::Assign(assign) if assign.op == AssignOp::Assign => {
                self.classify_value(&assign.right)
            }
            Expr::Seq(sequence) => sequence
                .exprs
                .last()
                .and_then(|expr| self.classify_value(expr)),
            Expr::Cond(cond) => {
                let consequent = self.classify_value(&cond.cons)?;
                (self.classify_value(&cond.alt) == Some(consequent)).then_some(consequent)
            }
            _ => None,
        }
    }

    fn classify_member(&self, member: &MemberExpr) -> Option<ScriptTraceValue> {
        if self.is_runtime_global_member(&Expr::Member(member.clone()), "document") {
            return Some(ScriptTraceValue::Document);
        }
        if self.allow_runtime_location
            && self.is_runtime_global_member(&Expr::Member(member.clone()), "location")
        {
            return Some(ScriptTraceValue::ScriptUrl);
        }
        if member_prop_is(&member.prop, "url")
            && matches!(strip_parens(&member.obj), Expr::MetaProp(meta) if meta.kind == MetaPropKind::ImportMeta)
        {
            return Some(ScriptTraceValue::ScriptUrl);
        }
        let object = self.classify_value(&member.obj)?;
        match (&member.prop, object) {
            (MemberProp::Ident(prop), ScriptTraceValue::Document)
                if prop.sym.as_str() == "currentScript" =>
            {
                Some(ScriptTraceValue::ScriptElement)
            }
            (MemberProp::Ident(prop), ScriptTraceValue::ScriptElement)
                if prop.sym.as_str() == "src" =>
            {
                Some(ScriptTraceValue::ScriptUrl)
            }
            (MemberProp::Computed(_), ScriptTraceValue::ScriptCollection) => {
                Some(ScriptTraceValue::ScriptElement)
            }
            _ => None,
        }
    }

    fn classify_call(&self, call: &CallExpr) -> Option<ScriptTraceValue> {
        let Callee::Expr(callee) = &call.callee else {
            return None;
        };
        let Expr::Member(member) = strip_parens(callee) else {
            return None;
        };
        if member_prop_is(&member.prop, "getElementsByTagName")
            && self.classify_value(&member.obj) == Some(ScriptTraceValue::Document)
            && single_string_argument(call) == Some("script")
        {
            return Some(ScriptTraceValue::ScriptCollection);
        }
        if !member_prop_is(&member.prop, "replace") {
            return None;
        }
        let receiver = self.classify_value(&member.obj)?;
        let (pattern, replacement) = regex_and_string_arguments(call)?;
        match (pattern, replacement, receiver) {
            ("^blob:", "", ScriptTraceValue::ScriptUrl)
            | ("#.*$", "", ScriptTraceValue::ScriptUrl)
            | (r"\?.*$", "", ScriptTraceValue::ScriptUrl) => Some(ScriptTraceValue::ScriptUrl),
            (r"\/[^\/]+$", "/", ScriptTraceValue::ScriptUrl) => {
                Some(ScriptTraceValue::ScriptDirectory)
            }
            _ => None,
        }
    }

    fn is_global_ident(&self, ident: &Ident, name: &str) -> bool {
        ident.sym.as_str() == name && ident.ctxt.outer() == self.unresolved_mark
    }

    fn is_runtime_global_member(&self, expr: &Expr, property: &str) -> bool {
        let Expr::Member(outer) = strip_parens(expr) else {
            return false;
        };
        if !member_prop_is(&outer.prop, property) {
            return false;
        }
        let Expr::Member(global) = strip_parens(&outer.obj) else {
            return false;
        };
        if !member_prop_is(&global.prop, "g") {
            return false;
        }
        matches!(strip_parens(&global.obj), Expr::Ident(runtime) if runtime.to_id() == self.runtime)
    }
}

impl Visit for ScriptPathTrace {
    fn visit_var_declarator(&mut self, declarator: &VarDeclarator) {
        if let Some(init) = declarator.init.as_deref() {
            init.visit_with(self);
            if let Pat::Ident(binding) = &declarator.name {
                self.set_binding(binding.id.to_id(), self.classify_value(init));
            }
        }
    }

    fn visit_assign_expr(&mut self, assign: &AssignExpr) {
        let assigned_value = (assign.op == AssignOp::Assign)
            .then(|| self.classify_value(&assign.right))
            .flatten();
        let public_path_fact = match &assign.left {
            AssignTarget::Simple(SimpleAssignTarget::Member(member))
                if assign.op == AssignOp::Assign
                    && member_prop_is(&member.prop, "p")
                    && matches!(strip_parens(&member.obj), Expr::Ident(runtime) if runtime.to_id() == self.runtime) =>
            {
                Some(self.classify_public_path(&assign.right))
            }
            _ => None,
        };
        assign.right.visit_with(self);
        match &assign.left {
            AssignTarget::Simple(SimpleAssignTarget::Ident(binding)) => {
                self.set_binding(binding.id.to_id(), assigned_value);
            }
            AssignTarget::Simple(SimpleAssignTarget::Member(_)) if public_path_fact.is_some() => {
                let fact = public_path_fact.expect("checked above");
                self.public_path_assignments.push(fact);
            }
            _ => {}
        }
    }

    fn visit_bin_expr(&mut self, bin: &swc_core::ecma::ast::BinExpr) {
        bin.left.visit_with(self);
        let previous = self.allow_runtime_location;
        if bin.op == BinaryOp::LogicalAnd
            && self.is_runtime_global_member(&bin.left, "importScripts")
        {
            self.allow_runtime_location = true;
        }
        bin.right.visit_with(self);
        self.allow_runtime_location = previous;
    }
}

fn member_prop_is(prop: &MemberProp, expected: &str) -> bool {
    matches!(prop, MemberProp::Ident(prop) if prop.sym.as_str() == expected)
}

fn is_empty_string(expr: &Expr) -> bool {
    matches!(strip_parens(expr), Expr::Lit(Lit::Str(value)) if value.value.as_str() == Some(""))
}

fn single_string_argument(call: &CallExpr) -> Option<&str> {
    let [argument] = call.args.as_slice() else {
        return None;
    };
    if argument.spread.is_some() {
        return None;
    }
    let Expr::Lit(Lit::Str(value)) = strip_parens(&argument.expr) else {
        return None;
    };
    value.value.as_str()
}

fn regex_and_string_arguments(call: &CallExpr) -> Option<(&str, &str)> {
    let [pattern, replacement] = call.args.as_slice() else {
        return None;
    };
    if pattern.spread.is_some() || replacement.spread.is_some() {
        return None;
    }
    let Expr::Lit(Lit::Regex(pattern)) = strip_parens(&pattern.expr) else {
        return None;
    };
    let Expr::Lit(Lit::Str(replacement)) = strip_parens(&replacement.expr) else {
        return None;
    };
    Some((pattern.exp.as_str(), replacement.value.as_str()?))
}

// ---------------------------------------------------------------------------
// Shared surface collection
// ---------------------------------------------------------------------------

/// Runtime-shaped facts collected in one pass over the whole input module.
///
struct RuntimeSurface {
    /// Identifiers bound to functions (declarations or fn/arrow initializers).
    function_names: HashSet<Id>,
    /// `obj.u = <expr>` assignments (webpack 5 JS chunk filename).
    u_assignments: HashMap<Id, Vec<Expr>>,
    /// `obj.miniCssF = <expr>` assignments (webpack 5 CSS chunk filename).
    mini_css_assignments: HashMap<Id, Vec<Expr>>,
    /// Other known webpack runtime property names assigned per object.
    other_runtime_props: HashMap<Id, HashSet<String>>,
    /// Literal ids from binding-correlated `X.e(id).then(X.bind(...))` /
    /// `X.t.bind(...)` async chunk load sites, across all scopes. Module
    /// factories receive the runtime under their own resolved parameter Id,
    /// so correlation is local to each generated continuation rather than to
    /// the bootstrap runtime binding.
    ensure_ids: BTreeSet<String>,
    /// Single-parameter functions whose body is `return X.p + <expr>` —
    /// webpack 4's `jsonpScriptSrc` shape. Stores (anchor object name,
    /// parsed template segments); dynamic templates store `None`.
    jsonp_script_src_fns: Vec<(Id, Option<Vec<Segment>>)>,
    unresolved_mark: Mark,
}

impl RuntimeSurface {
    fn collect(module: &Module, unresolved_mark: Mark) -> Self {
        let mut surface = Self {
            function_names: HashSet::new(),
            u_assignments: HashMap::new(),
            mini_css_assignments: HashMap::new(),
            other_runtime_props: HashMap::new(),
            ensure_ids: BTreeSet::new(),
            jsonp_script_src_fns: Vec::new(),
            unresolved_mark,
        };
        module.visit_with(&mut surface);
        surface
    }
}

impl Visit for RuntimeSurface {
    fn visit_fn_decl(&mut self, decl: &FnDecl) {
        self.function_names.insert(decl.ident.to_id());
        if let Some(parts) = function_parts(&decl.function) {
            self.collect_jsonp_script_src(parts);
        }
        decl.visit_children_with(self);
    }

    fn visit_fn_expr(&mut self, fn_expr: &swc_core::ecma::ast::FnExpr) {
        // Minifiers inline webpack 4's `jsonpScriptSrc` into an immediately
        // invoked function expression at the `script.src` assignment, so the
        // shape must be recognized wherever a function appears.
        if let Some(parts) = function_parts(&fn_expr.function) {
            self.collect_jsonp_script_src(parts);
        }
        fn_expr.visit_children_with(self);
    }

    fn visit_arrow_expr(&mut self, arrow: &swc_core::ecma::ast::ArrowExpr) {
        if let Some(parts) = arrow_parts(arrow) {
            self.collect_jsonp_script_src(parts);
        }
        arrow.visit_children_with(self);
    }

    fn visit_var_declarator(&mut self, declarator: &VarDeclarator) {
        if let Pat::Ident(binding) = &declarator.name {
            if let Some(init) = declarator.init.as_deref() {
                if matches!(strip_parens(init), Expr::Fn(_) | Expr::Arrow(_)) {
                    self.function_names.insert(binding.id.to_id());
                }
            }
        }
        declarator.visit_children_with(self);
    }

    fn visit_assign_expr(&mut self, assign: &AssignExpr) {
        if let AssignTarget::Simple(SimpleAssignTarget::Member(member)) = &assign.left {
            if let (Expr::Ident(obj), MemberProp::Ident(prop)) = (&*member.obj, &member.prop) {
                let object = obj.to_id();
                let value = (*assign.right).clone();
                match prop.sym.as_str() {
                    "u" => self.u_assignments.entry(object).or_default().push(value),
                    "miniCssF" => self
                        .mini_css_assignments
                        .entry(object)
                        .or_default()
                        .push(value),
                    name if is_webpack_runtime_property(name) => {
                        self.other_runtime_props
                            .entry(object)
                            .or_default()
                            .insert(name.to_string());
                    }
                    _ => {}
                }
            }
        }
        assign.visit_children_with(self);
    }

    fn visit_call_expr(&mut self, call: &CallExpr) {
        self.collect_ensure_ids(call);
        call.visit_children_with(self);
    }
}

impl RuntimeSurface {
    /// Match webpack 4's script-path function: one parameter, single
    /// `return X.p + <template>` body. The template outcome is recorded even
    /// when dynamic so ambiguity between candidates can fail closed.
    fn collect_jsonp_script_src(&mut self, (param, body): (Id, &Expr)) {
        let Expr::Bin(bin) = body else {
            return;
        };
        if bin.op != BinaryOp::Add {
            return;
        }
        // The leftmost leaf of the concatenation must be `X.p`.
        let mut leftmost = strip_parens(&bin.left);
        let mut rest = vec![&*bin.right];
        while let Expr::Bin(inner) = leftmost {
            if inner.op != BinaryOp::Add {
                return;
            }
            rest.push(&*inner.right);
            leftmost = strip_parens(&inner.left);
        }
        let Expr::Member(MemberExpr { obj, prop, .. }) = leftmost else {
            return;
        };
        let (Expr::Ident(anchor), MemberProp::Ident(prop)) = (&**obj, prop) else {
            return;
        };
        if prop.sym.as_str() != "p" {
            return;
        }
        let mut segments = Vec::new();
        let static_template = rest
            .into_iter()
            .rev()
            .all(|part| parse_segments(part, &param, &mut segments));
        self.jsonp_script_src_fns
            .push((anchor.to_id(), static_template.then_some(segments)));
    }

    /// Collect literal chunk ids from generated `X.e(id).then(X.bind(...))`
    /// and `Promise.all([X.e(a), ...]).then(X.bind(...))` load sites. The
    /// continuation must use the same resolved binding as every `.e` receiver.
    fn collect_ensure_ids(&mut self, call: &CallExpr) {
        let Callee::Expr(callee) = &call.callee else {
            return;
        };
        let Expr::Member(MemberExpr { obj, prop, .. }) = strip_parens(callee) else {
            return;
        };
        let MemberProp::Ident(prop) = prop else {
            return;
        };
        if prop.sym.as_str() != "then" {
            return;
        }
        let Some(callback_runtime) = call
            .args
            .first()
            .filter(|argument| argument.spread.is_none())
            .and_then(|argument| bound_runtime_id(&argument.expr))
        else {
            return;
        };
        let receiver = strip_parens(obj);
        if let Some((runtime, id)) = ensure_call_literal_id(receiver) {
            if runtime == callback_runtime {
                self.ensure_ids.insert(id);
            }
            return;
        }
        // Promise.all([X.e(a), X.e(b), ...])
        let Expr::Call(promise_all) = receiver else {
            return;
        };
        let Callee::Expr(all_callee) = &promise_all.callee else {
            return;
        };
        let Expr::Member(MemberExpr { obj, prop, .. }) = strip_parens(all_callee) else {
            return;
        };
        let (Expr::Ident(promise), MemberProp::Ident(prop)) = (strip_parens(obj), prop) else {
            return;
        };
        if promise.sym.as_str() != "Promise"
            || promise.ctxt.outer() != self.unresolved_mark
            || prop.sym.as_str() != "all"
        {
            return;
        }
        let [argument] = promise_all.args.as_slice() else {
            return;
        };
        if argument.spread.is_some() {
            return;
        }
        let Expr::Array(elements) = strip_parens(&argument.expr) else {
            return;
        };
        let mut ids = Vec::with_capacity(elements.elems.len());
        for element in &elements.elems {
            let Some(element) = element else {
                return;
            };
            if element.spread.is_some() {
                return;
            }
            let Some((runtime, id)) = ensure_call_literal_id(strip_parens(&element.expr)) else {
                return;
            };
            if runtime != callback_runtime {
                return;
            }
            ids.push(id);
        }
        self.ensure_ids.extend(ids);
    }
}

/// Match `X.e(<literal id>)` and return the id as a string.
fn ensure_call_literal_id(expr: &Expr) -> Option<(Id, String)> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Callee::Expr(callee) = &call.callee else {
        return None;
    };
    let Expr::Member(MemberExpr { obj, prop, .. }) = strip_parens(callee) else {
        return None;
    };
    let MemberProp::Ident(prop) = prop else {
        return None;
    };
    if prop.sym.as_str() != "e" {
        return None;
    }
    let Expr::Ident(runtime) = strip_parens(obj) else {
        return None;
    };
    let [argument] = call.args.as_slice() else {
        return None;
    };
    if argument.spread.is_some() {
        return None;
    }
    match strip_parens(&argument.expr) {
        Expr::Lit(Lit::Str(value)) => value
            .value
            .as_str()
            .map(|value| (runtime.to_id(), value.to_string())),
        Expr::Lit(Lit::Num(value)) => {
            integer_string(value.value).map(|value| (runtime.to_id(), value))
        }
        _ => None,
    }
}

/// Match webpack's generated `.then(runtime.bind(...))` and
/// `.then(runtime.t.bind(...))` continuations. Correlating the continuation
/// with the `.e(id)` receiver keeps authored `client.e(id).then(...)` calls
/// from becoming invented chunk ids while still accepting factory-local
/// runtime parameters.
fn bound_runtime_id(expr: &Expr) -> Option<Id> {
    let Expr::Call(call) = strip_parens(expr) else {
        return None;
    };
    let Callee::Expr(callee) = &call.callee else {
        return None;
    };
    let Expr::Member(bind) = strip_parens(callee) else {
        return None;
    };
    if !member_prop_is(&bind.prop, "bind") {
        return None;
    }
    let runtime = match strip_parens(&bind.obj) {
        Expr::Ident(runtime) => runtime,
        Expr::Member(member) if member_prop_is(&member.prop, "t") => {
            let Expr::Ident(runtime) = strip_parens(&member.obj) else {
                return None;
            };
            runtime
        }
        _ => return None,
    };
    let this_argument = call.args.first()?;
    if this_argument.spread.is_some() {
        return None;
    }
    match strip_parens(&this_argument.expr) {
        Expr::Lit(Lit::Null(_)) => Some(runtime.to_id()),
        Expr::Ident(this_value) if this_value.to_id() == runtime.to_id() => Some(runtime.to_id()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Per-format extraction
// ---------------------------------------------------------------------------

fn extract_webpack5(module: &Module, unresolved_mark: Mark) -> Option<ChunkEnumeration> {
    let surface = RuntimeSurface::collect(module, unresolved_mark);

    // Candidate runtime objects: a function-valued binding carrying a `.u`
    // assignment plus at least two other known runtime property assignments.
    // Anything ambiguous (zero or multiple candidates, or a candidate with
    // conflicting `.u` assignments) fails closed to `None`.
    let candidates: Vec<&Id> = surface
        .u_assignments
        .iter()
        .filter(|(object, assignments)| {
            surface.function_names.contains(*object)
                && surface
                    .other_runtime_props
                    .get(*object)
                    .is_some_and(|props| props.len() >= 2)
                && assignments
                    .iter()
                    .all(|value| matches!(strip_parens(value), Expr::Fn(_) | Expr::Arrow(_)))
        })
        .map(|(object, _)| object)
        .collect();
    let [candidate] = candidates.as_slice() else {
        return None;
    };
    let candidate = (*candidate).clone();
    let [u_expr] = surface.u_assignments[&candidate].as_slice() else {
        return None;
    };

    let mut assets = Vec::new();
    assets.push(match parse_filename_fn(u_expr) {
        Some(segments) => enumerate_asset(ChunkAssetKind::Js, segments, &surface.ensure_ids),
        None => dynamic_asset(ChunkAssetKind::Js),
    });

    if let Some(css_assignments) = surface.mini_css_assignments.get(&candidate) {
        assets.push(match css_assignments.as_slice() {
            // CSS ids come only from the template's own lookup tables:
            // `.e(id)` load sites do not prove a chunk has a CSS asset.
            [css_expr] => match parse_filename_fn(css_expr) {
                Some(segments) => enumerate_asset(ChunkAssetKind::Css, segments, &BTreeSet::new()),
                None => dynamic_asset(ChunkAssetKind::Css),
            },
            _ => dynamic_asset(ChunkAssetKind::Css),
        });
    }

    Some(ChunkEnumeration {
        public_path: ScriptPathTrace::collect(module, candidate, unresolved_mark),
        assets,
        relative_imports: Vec::new(),
    })
}

fn extract_webpack4(module: &Module, unresolved_mark: Mark) -> Option<ChunkEnumeration> {
    let surface = RuntimeSurface::collect(module, unresolved_mark);

    // Candidate: exactly one distinct `return X.p + <template>` function
    // whose anchor object is a function-valued binding (webpack 4's
    // `__webpack_require__`). Duplicate identical (anchor, template) pairs
    // collapse; genuinely different candidates fail closed.
    let mut candidates: Vec<(Id, Option<Vec<Segment>>)> = Vec::new();
    for candidate in &surface.jsonp_script_src_fns {
        if !surface.function_names.contains(&candidate.0) {
            continue;
        }
        if !candidates.contains(candidate) {
            candidates.push(candidate.clone());
        }
    }
    let [(anchor, template)] = candidates.as_slice() else {
        return None;
    };

    let asset = match template {
        Some(segments) => {
            enumerate_asset(ChunkAssetKind::Js, segments.clone(), &surface.ensure_ids)
        }
        None => dynamic_asset(ChunkAssetKind::Js),
    };

    Some(ChunkEnumeration {
        public_path: ScriptPathTrace::collect(module, anchor.clone(), unresolved_mark),
        assets: vec![asset],
        relative_imports: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Literal relative ESM specifier collection (format-independent)
// ---------------------------------------------------------------------------

/// Collect literal *relative* module specifiers the input imports statically
/// or via `import()`. For native ESM code-splitting (Vite/Rollup/rolldown/
/// esbuild output, which wakaru surfaces as scope-hoisted or plain), each
/// such specifier is the exact relative URL of a sibling chunk.
///
/// Fail-closed: only string-literal specifiers are collected; `import(expr)`,
/// `import("./" + x)`, and non-relative specifiers (bare packages, absolute
/// URLs, protocol-relative) are ignored. Deduplicated by `(specifier, kind)`;
/// order is source order, so a URL used both eagerly and lazily retains both
/// facts.
pub fn collect_relative_import_specifiers(module: &Module) -> Vec<RelativeImport> {
    let mut collector = SpecifierCollector::default();
    // Top-level static import/export-from declarations.
    for item in &module.body {
        if let ModuleItem::ModuleDecl(decl) = item {
            match decl {
                ModuleDecl::Import(import) => {
                    collector.record(import.src.value.as_str(), RelativeImportKind::Import);
                }
                ModuleDecl::ExportNamed(export) => {
                    if let Some(src) = &export.src {
                        collector.record(src.value.as_str(), RelativeImportKind::ExportFrom);
                    }
                }
                ModuleDecl::ExportAll(export) => {
                    collector.record(export.src.value.as_str(), RelativeImportKind::ExportFrom);
                }
                _ => {}
            }
        }
    }
    // Dynamic `import(...)` calls anywhere in the module.
    module.visit_with(&mut collector);
    collector.into_imports()
}

fn is_relative_specifier(specifier: &str) -> bool {
    specifier.starts_with("./") || specifier.starts_with("../")
}

#[derive(Default)]
struct SpecifierCollector {
    seen: HashSet<(String, RelativeImportKind)>,
    imports: Vec<RelativeImport>,
}

impl SpecifierCollector {
    fn record(&mut self, specifier: Option<&str>, kind: RelativeImportKind) {
        let Some(specifier) = specifier else {
            return;
        };
        if !is_relative_specifier(specifier) {
            return;
        }
        if self.seen.insert((specifier.to_string(), kind)) {
            self.imports.push(RelativeImport {
                specifier: specifier.to_string(),
                kind,
            });
        }
    }

    fn into_imports(self) -> Vec<RelativeImport> {
        self.imports
    }
}

impl Visit for SpecifierCollector {
    fn visit_call_expr(&mut self, call: &CallExpr) {
        if matches!(call.callee, Callee::Import(_)) {
            if let [argument] = call.args.as_slice() {
                if argument.spread.is_none() {
                    if let Expr::Lit(Lit::Str(value)) = strip_parens(&argument.expr) {
                        self.record(value.value.as_str(), RelativeImportKind::DynamicImport);
                    }
                }
            }
        }
        call.visit_children_with(self);
    }
}
