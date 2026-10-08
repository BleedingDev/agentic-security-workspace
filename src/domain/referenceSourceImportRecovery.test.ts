import { describe, expect, it } from "vitest";

import { parseReferenceSourceImports } from "./referenceSourceImportParsing.js";

const recoverySource = [
  "const missing;",
  "let duplicate;",
  "let duplicate;",
  "let duplicate;",
].join("\n");

const recoveryReasons = [
  "Missing initializer in const declaration. (1:13)",
  "Identifier 'duplicate' has already been declared. (3:4)",
  "Identifier 'duplicate' has already been declared. (4:4)",
];

const moduleSource = [
  'import "./static.js";',
  'export { value } from "./named.js";',
  'export * from "./all.js";',
  'require("./common.cjs");',
  'require.resolve("node:fs");',
  'import("./lazy.js");',
  "import(moduleName);",
].join("\n");

const moduleRelationships = [
  ["./static.js", "imports", "internal", "parsed"],
  ["./named.js", "imports", "internal", "parsed"],
  ["./all.js", "imports", "internal", "parsed"],
  ["./common.cjs", "requires", "internal", "parsed"],
  ["node:fs", "requires", "external", "parsed"],
  ["./lazy.js", "imports", "internal", "parsed"],
  ["<dynamic-import>", "imports", "unknown", "partial"],
].map(([to, kind, resolution, parse_state]) => ({
  to,
  kind,
  resolution,
  parse_state,
}));

const parseSource = (path: string, source: string, language: string | null) =>
  parseReferenceSourceImports(path, new TextEncoder().encode(source), language);

describe.each([
  ["main.js", "JavaScript"],
  ["main.jsx", "JSX"],
  ["main.ts", "TypeScript"],
  ["main.tsx", "TSX"],
])("recovered reference imports in %s", (path, language) => {
  it("retains every diagnostic and marks recovered relationships partial", () => {
    const result = parseSource(
      path,
      `${recoverySource}\n${moduleSource}`,
      language,
    );

    expect(result.parse_failures).toEqual(
      recoveryReasons.map((reason) => ({ path, parser: "babel", reason })),
    );
    expect(result.relationships).toEqual(
      moduleRelationships.map((relationship) => ({
        ...relationship,
        from_path: path,
        parse_state: "partial",
      })),
    );
  });

  it("preserves valid and computed relationships without diagnostics", () => {
    expect(parseSource(path, moduleSource, language)).toEqual({
      relationships: moduleRelationships.map((relationship) => ({
        ...relationship,
        from_path: path,
      })),
      parse_failures: [],
    });
  });

  it("reports recovery diagnostics when there are no relationships", () => {
    expect(parseSource(path, recoverySource, language)).toEqual({
      relationships: [],
      parse_failures: recoveryReasons.map((reason) => ({
        path,
        parser: "babel",
        reason,
      })),
    });
  });

  it("returns no relationships when parsing throws", () => {
    expect(
      parseSource(path, 'import "./before.js";\nconst = ;', language),
    ).toEqual({
      relationships: [],
      parse_failures: [
        { path, parser: "babel", reason: "Unexpected token (2:6)" },
      ],
    });
  });
});

it.each(["main.ts", "types.d.ts", "types.d.mts", "types.d.cts"])(
  "marks recovered TypeScript declarations partial in %s",
  (path) => {
    const result = parseSource(
      path,
      [
        "let duplicate: number;",
        "let duplicate: number;",
        'import dep = require("node:path");',
        'declare module "ambient-package" {}',
      ].join("\n"),
      "TypeScript",
    );
    expect(result).toEqual({
      relationships: [
        {
          from_path: path,
          to: "node:path",
          kind: "requires",
          resolution: "external",
          parse_state: "partial",
        },
        {
          from_path: path,
          to: "ambient-package",
          kind: "declares-module",
          resolution: "unknown",
          parse_state: "partial",
        },
      ],
      parse_failures: [
        {
          path,
          parser: "babel",
          reason: "Identifier 'duplicate' has already been declared. (2:4)",
        },
      ],
    });
  },
);

it("keeps unsupported source languages outside Babel parsing", () => {
  expect(parseSource("main.py", recoverySource, "Python")).toEqual({
    relationships: [],
    parse_failures: [],
  });
});
