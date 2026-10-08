import { createRequire } from "node:module";
import { join } from "node:path";
import { ensureNodeTool } from "../lib/runner.mjs";
import { matchesAnyForm, prewarmNormalize } from "../lib/compare.mjs";

let typescript;
function tsApi() {
  typescript ??= createRequire(join(ensureNodeTool("typescript-5.9.3", ["typescript@5.9.3"]), "package.json"))("typescript");
  return typescript;
}

// The shared identifier normalizer handles ordinary bindings, not #names.
// Use lexical class scopes, including references captured by nested classes.
// A nested class's heritage expression is evaluated in its outer private scope.
function canonicalizePrivateNames(ts, file) {
  let nextName = 0;
  let invalid = false;
  const scopes = [];
  const transformed = ts.transform(file, [(context) => {
    function visit(node) {
      if (ts.isClassDeclaration(node) || ts.isClassExpression(node)) {
        const scope = new Map();
        for (const member of node.members) {
          if (member.name && ts.isPrivateIdentifier(member.name) && !scope.has(member.name.text)) {
            scope.set(member.name.text, `#__private${nextName++}`);
          }
        }
        scopes.push(scope);
        const result = ts.visitEachChild(node, (child) => {
          if (!ts.isHeritageClause(child)) return visit(child);
          scopes.pop();
          const heritage = visit(child);
          scopes.push(scope);
          return heritage;
        }, context);
        scopes.pop();
        return result;
      }
      if (ts.isPrivateIdentifier(node)) {
        for (let i = scopes.length - 1; i >= 0; i--) {
          const name = scopes[i].get(node.text);
          if (name) return ts.factory.createPrivateIdentifier(name);
        }
        // Do not turn an unbound private name into a generated binding.
        invalid = true;
        return node;
      }
      return ts.visitEachChild(node, visit, context);
    }
    return visit;
  }]);
  const result = transformed.transformed[0];
  transformed.dispose();
  return invalid ? null : result;
}

// Compare the complete recovered module, not just the presence of async/class
// syntax. Canonicalize private bindings and export spelling/placement. A CJS round trip can
// emit `function f() {}; export { f }` instead of `export function f() {}`.
// Keep local export targets distinct from their public names for alpha-renaming.
function moduleParts(code) {
  const ts = tsApi();
  const parsed = ts.createSourceFile("module.js", code, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  if (parsed.parseDiagnostics.length) return null;
  const file = canonicalizePrivateNames(ts, parsed);
  if (!file) return null;
  const printer = ts.createPrinter();
  const body = [], exports = [], tslib = [];
  for (let statement of file.statements) {
    if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier)) {
      statement = ts.factory.updateImportDeclaration(statement, statement.modifiers,
        statement.importClause, ts.factory.createStringLiteral(statement.moduleSpecifier.text), statement.attributes);
    }
    if (ts.isImportDeclaration(statement) && statement.moduleSpecifier.text === "tslib") {
      tslib.push(printer.printNode(ts.EmitHint.Unspecified, statement, file));
    } else if (ts.isExportDeclaration(statement) && !statement.moduleSpecifier
      && statement.exportClause && ts.isNamedExports(statement.exportClause)) {
      exports.push(...statement.exportClause.elements);
    } else if (ts.isVariableStatement(statement)
      && statement.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword)) {
      // Keep public names outside the alpha-renamed declaration, just as for
      // functions/classes. Do not fold initializers or assignment statements.
      if (!statement.declarationList.declarations.every((decl) => ts.isIdentifier(decl.name))) return null;
      exports.push(...statement.declarationList.declarations.map((decl) =>
        ts.factory.createExportSpecifier(false, undefined, decl.name)));
      body.push(printer.printNode(ts.EmitHint.Unspecified, ts.factory.replaceModifiers(
        statement, statement.modifiers.filter((m) => m.kind !== ts.SyntaxKind.ExportKeyword),
      ), file));
    } else if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement))
      && statement.name && statement.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword)) {
      // Preserve default declarations exactly: splitting them can change a
      // live default binding into a snapshot. Any alternate form is snippet-local.
      if (statement.modifiers.some((m) => m.kind === ts.SyntaxKind.DefaultKeyword)) {
        body.push(printer.printNode(ts.EmitHint.Unspecified, statement, file));
        continue;
      }
      exports.push(ts.factory.createExportSpecifier(false, undefined, statement.name));
      body.push(printer.printNode(ts.EmitHint.Unspecified, ts.factory.replaceModifiers(
        statement, statement.modifiers.filter((m) => m.kind !== ts.SyntaxKind.ExportKeyword),
      ), file));
    } else {
      body.push(printer.printNode(ts.EmitHint.Unspecified, statement, file));
    }
  }
  exports.sort((a, b) => a.name.text.localeCompare(b.name.text, "en"));
  if (exports.length) body.push(printer.printNode(ts.EmitHint.Unspecified,
    ts.factory.createExportDeclaration(undefined, false, ts.factory.createNamedExports(exports)), file));
  return { body: body.join("\n"), tslib: tslib.join("\n") };
}

export function comparisonInputs(snippet, recovered) {
  const actual = moduleParts(recovered);
  if (!actual) return { recovered, forms: [] };
  // Helper imports are injected by the compiler and may survive after their
  // calls disappear. Include those exact imports in the expected module too;
  // never erase helper calls, declarations, or any other program statements.
  // Other imports (e.g. the interop provider) must match the source normally.
  const prefix = actual.tslib + "\n";
  const forms = [snippet.source, ...(snippet.acceptForms ?? [])]
    .map(moduleParts).filter(Boolean).map((form) => prefix + form.body);
  return { recovered: prefix + actual.body, forms };
}

export function validateModuleRecovery({ snippet, recovered }) {
  const input = comparisonInputs(snippet, recovered);
  const success = matchesAnyForm(input.recovered, input.forms);
  return {
    recovered: success,
    notes: success ? "complete module matches an accepted source form" : "module differs from accepted source forms",
  };
}

export async function prewarmModuleRecovery(rows) {
  await prewarmNormalize(rows.flatMap(({ snippet, recovered }) => {
    if (recovered == null) return [];
    const input = comparisonInputs(snippet, recovered);
    return [input.recovered, ...input.forms];
  }), { rename: true });
}
