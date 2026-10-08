use swc_core::common::{Mark, SyntaxContext};
use swc_core::ecma::ast::{Expr, Ident, Lit, Module, Number, UnaryExpr, UnaryOp};
use swc_core::ecma::visit::{VisitMut, VisitMutWith};

use super::eval_utils::module_blocks_global_reference;
use super::rename_utils::module_declares_binding_named;
use crate::utils::paren::strip_parens;

/// Rewrites `void <number>` back to `undefined`.
///
/// The synthesized identifier is a free reference to the global, so the rule
/// skips a module that declares an `undefined` binding anywhere, and, per the
/// dynamic-scope policy in `docs/rewrite-assumptions.md`, a module containing
/// `with` or a direct `eval` that could bind the name at runtime.
pub struct RemoveVoid {
    unresolved_ctxt: SyntaxContext,
}

impl RemoveVoid {
    pub fn new(unresolved_mark: Mark) -> Self {
        Self {
            unresolved_ctxt: SyntaxContext::empty().apply_mark(unresolved_mark),
        }
    }

    pub fn should_run(module: &Module) -> bool {
        !declares_undefined_binding(module) && !module_blocks_global_reference(module, "undefined")
    }
}

/// Whether any binding in the module is spelled `undefined`.
fn declares_undefined_binding(module: &Module) -> bool {
    module_declares_binding_named(module, "undefined")
}

impl VisitMut for RemoveVoid {
    fn visit_mut_unary_expr(&mut self, expr: &mut UnaryExpr) {
        if expr.op == UnaryOp::Delete {
            return;
        }
        expr.visit_mut_children_with(self);
    }

    fn visit_mut_expr(&mut self, expr: &mut Expr) {
        expr.visit_mut_children_with(self);

        if let Expr::Unary(UnaryExpr { op, arg, span }) = expr {
            if *op == UnaryOp::Void && is_numeric_literal(strip_parens(arg)) {
                *expr = Expr::Ident(Ident::new("undefined".into(), *span, self.unresolved_ctxt));
            }
        }
    }
}

fn is_numeric_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Lit(Lit::Num(_)))
}

/// Gives context-less `undefined` references the global's context, or spells
/// them `void 0` when a binding named `undefined` exists anywhere in the
/// module so that binding cannot capture them. Rules that synthesize
/// `undefined` deep inside their decoders call this once per module instead of
/// threading the unresolved mark through every helper.
pub(crate) fn finalize_synthesized_undefined(module: &mut Module, unresolved_mark: Mark) {
    struct Finalizer {
        shadowed: bool,
        unresolved_ctxt: SyntaxContext,
    }

    impl VisitMut for Finalizer {
        fn visit_mut_expr(&mut self, expr: &mut Expr) {
            expr.visit_mut_children_with(self);
            let Expr::Ident(id) = expr else {
                return;
            };
            if id.sym != "undefined" || id.ctxt != SyntaxContext::empty() {
                return;
            }
            if self.shadowed {
                *expr = Expr::Unary(UnaryExpr {
                    span: id.span,
                    op: UnaryOp::Void,
                    arg: Box::new(Expr::Lit(Lit::Num(Number {
                        span: id.span,
                        value: 0.0,
                        raw: None,
                    }))),
                });
            } else {
                id.ctxt = self.unresolved_ctxt;
            }
        }
    }

    let shadowed = declares_undefined_binding(module);
    module.visit_mut_with(&mut Finalizer {
        shadowed,
        unresolved_ctxt: SyntaxContext::empty().apply_mark(unresolved_mark),
    });
}

#[cfg(test)]
mod tests {
    use swc_core::common::{sync::Lrc, Globals, SourceMap, DUMMY_SP, GLOBALS};
    use swc_core::ecma::ast::{ExprStmt, ModuleItem, Stmt};
    use swc_core::ecma::transforms::base::resolver;

    use super::*;

    fn module_with_synthesized_undefined(source: &str) -> (Module, Mark) {
        let cm: Lrc<SourceMap> = Default::default();
        let mut module = crate::unpacker::parse_es_module(source, "fixture.js", cm)
            .expect("fixture should parse");
        let unresolved_mark = Mark::new();
        module.visit_mut_with(&mut resolver(unresolved_mark, Mark::new(), false));
        module.body.push(ModuleItem::Stmt(Stmt::Expr(ExprStmt {
            span: DUMMY_SP,
            expr: Box::new(Expr::Ident(Ident::new_no_ctxt(
                "undefined".into(),
                DUMMY_SP,
            ))),
        })));
        (module, unresolved_mark)
    }

    fn last_expr(module: &Module) -> &Expr {
        let Some(ModuleItem::Stmt(Stmt::Expr(stmt))) = module.body.last() else {
            panic!("expected a trailing expression statement");
        };
        &stmt.expr
    }

    #[test]
    fn synthesized_undefined_gets_the_unresolved_mark() {
        GLOBALS.set(&Globals::new(), || {
            let (mut module, unresolved_mark) = module_with_synthesized_undefined("use(x);");
            finalize_synthesized_undefined(&mut module, unresolved_mark);
            let Expr::Ident(id) = last_expr(&module) else {
                panic!("expected the identifier to stay");
            };
            assert_eq!(id.ctxt.outer(), unresolved_mark);
        });
    }

    #[test]
    fn synthesized_undefined_becomes_void_when_a_binding_spells_it() {
        GLOBALS.set(&Globals::new(), || {
            let (mut module, unresolved_mark) =
                module_with_synthesized_undefined("function f(undefined) { return undefined; }");
            finalize_synthesized_undefined(&mut module, unresolved_mark);
            assert!(
                matches!(
                    last_expr(&module),
                    Expr::Unary(UnaryExpr {
                        op: UnaryOp::Void,
                        ..
                    })
                ),
                "expected void 0, got {:?}",
                last_expr(&module)
            );
        });
    }
}
