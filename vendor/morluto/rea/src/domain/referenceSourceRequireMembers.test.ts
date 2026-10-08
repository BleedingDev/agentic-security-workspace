import { describe, expect, it } from "vitest";

import { parseReferenceSourceImports } from "./referenceSourceImportParsing.js";

const parseRequire = (source: string) =>
  parseReferenceSourceImports(
    "main.cjs",
    new TextEncoder().encode(source),
    "JavaScript",
  );

const knownCallees = [
  "require",
  "require.resolve",
  'require["resolve"]',
  "require.main",
  'require["main"]',
];

describe("historical source require member names", () => {
  it.each([
    ...knownCallees,
    'require[("resolve")]',
    'require["res\\u006flve"]',
  ])(
    "retains a literal dependency for %s beside computed members",
    (callee) => {
      expect(
        parseRequire(
          `require[resolve]("./ignored.js"); ${callee}("./dep.js");`,
        ),
      ).toEqual({
        relationships: [
          {
            from_path: "main.cjs",
            to: "./dep.js",
            kind: "requires",
            resolution: "internal",
            parse_state: "parsed",
          },
        ],
        parse_failures: [],
      });
    },
  );

  it.each([
    'require[resolve]("./dep.js");',
    'require[main]("./dep.js");',
    'const resolve = "toString"; require[resolve]("./dep.js");',
    'const main = "toString"; require[main]("./dep.js");',
    'const resolve = "resolve"; require[resolve]("./dep.js");',
    'const main = "main"; require[main]("./dep.js");',
    'const method = "resolve"; require[method]("./dep.js");',
    'require[`resolve`]("./dep.js");',
    'require["re" + "solve"]("./dep.js");',
  ])("does not infer computed member values in %s", (source) => {
    expect(parseRequire(source)).toEqual({
      relationships: [],
      parse_failures: [],
    });
  });

  it.each([
    "require?.",
    "require.resolve?.",
    'require["resolve"]?.',
    "require.main?.",
    'require["main"]?.',
    "require?.resolve",
    'require?.["resolve"]',
    "require?.[resolve]",
    "require?.main",
    'require?.["main"]',
    "require?.[main]",
    "(require?.resolve)",
    "require.resolve.call",
    "require.main.require",
    "module.require",
    "other.resolve",
    "other.require",
    "require.toString",
    'require["toString"]',
  ])("leaves the unrecognized callee %s unchanged", (callee) => {
    expect(parseRequire(`${callee}("./dep.js");`)).toEqual({
      relationships: [],
      parse_failures: [],
    });
  });

  it.each(knownCallees)("does not guess unknown arguments to %s", (callee) => {
    expect(
      parseRequire(
        [
          'const target = "./dep.js";',
          `${callee}(target);`,
          `${callee}(\`./dep.js\`);`,
          `${callee}(getTarget());`,
          `${callee}();`,
        ].join("\n"),
      ),
    ).toEqual({ relationships: [], parse_failures: [] });
  });
});
