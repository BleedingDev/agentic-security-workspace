import { describe, expect, it } from "vitest";

import { inferJsonShape } from "./jsonShape.js";

describe("inferJsonShape", () => {
  it("retains paths and types without retaining JSON values", () => {
    const shape = inferJsonShape(
      JSON.stringify({
        token: "super-secret",
        users: [
          { id: 1, active: true },
          { id: "second-secret", active: false },
        ],
        optional: null,
      }),
    );

    expect(shape).toMatchObject({
      root_type: "object",
      properties: expect.arrayContaining([
        { path: "/token", types: ["string"], observations: 1 },
        {
          path: "/users/*/id",
          types: ["number", "string"],
          observations: 2,
        },
        {
          path: "/users/*/active",
          types: ["boolean"],
          observations: 2,
        },
      ]),
    });
    expect(JSON.stringify(shape)).not.toContain("super-secret");
    expect(JSON.stringify(shape)).not.toContain("second-secret");
  });

  it("rejects malformed JSON", () => {
    expect(inferJsonShape("not-json")).toBeNull();
  });

  it.each([
    ["é", "e\u0301", "e\u0341"],
    ["é", "e\u0341", "e\u0301"],
    ["e\u0301", "é", "e\u0341"],
    ["e\u0301", "e\u0341", "é"],
    ["e\u0341", "é", "e\u0301"],
    ["e\u0341", "e\u0301", "é"],
  ])("orders distinct collation-equal paths from %j, %j, %j", (...names) => {
    for (const name of names) expect(name.localeCompare("é")).toBe(0);
    const values: Readonly<Record<string, unknown>> = {
      é: 1,
      "e\u0301": "value",
      "e\u0341": null,
    };
    const shape = inferJsonShape(
      JSON.stringify(
        Object.fromEntries(names.map((name) => [name, values[name]])),
      ),
    );

    expect(shape).toEqual({
      root_type: "object",
      node_count: 4,
      max_depth_observed: 1,
      properties: [
        { path: "/e\u0301", types: ["string"], observations: 1 },
        { path: "/e\u0341", types: ["null"], observations: 1 },
        { path: "/é", types: ["number"], observations: 1 },
      ],
    });
  });

  it.each([false, true])(
    "preserves nested types, counts, depth and pointer names with reversed keys: %s",
    (reverse) => {
      const object = (number: number, value: unknown) => {
        const entries = [
          ["é/~", number],
          ["e\u0301/~", value],
        ];
        return Object.fromEntries(reverse ? entries.reverse() : entries);
      };
      const shape = inferJsonShape(
        JSON.stringify({ nested: [object(1, null), object(2, [null])] }),
      );

      expect(shape).toEqual({
        root_type: "object",
        node_count: 9,
        max_depth_observed: 4,
        properties: [
          { path: "/nested", types: ["array"], observations: 1 },
          {
            path: "/nested/*/e\u0301~1~0",
            types: ["array", "null"],
            observations: 2,
          },
          { path: "/nested/*/é~1~0", types: ["number"], observations: 2 },
        ],
      });
    },
  );

  it("preserves the existing locale order for paths that do not tie", () => {
    const names = ["z", "é", "A", "_", "a"];
    const expected = names
      .map((name) => `/${name}`)
      .sort((left, right) => left.localeCompare(right));
    expect(new Set(expected).size).toBe(names.length);
    for (const [index, left] of expected.entries())
      for (const right of expected.slice(index + 1))
        expect(left.localeCompare(right)).not.toBe(0);
    const shape = inferJsonShape(
      JSON.stringify(Object.fromEntries(names.map((name) => [name, true]))),
    );

    expect(shape?.properties.map(({ path }) => path)).toEqual(expected);
  });

  it("retains every parsed property beyond the former shape-node limit", () => {
    const content = Object.fromEntries(
      Array.from({ length: 5_001 }, (_, index) => [`field_${index}`, index]),
    );
    const shape = inferJsonShape(JSON.stringify(content));

    expect(shape?.properties).toHaveLength(5_001);
    expect(shape?.node_count).toBe(5_002);
    expect(shape?.properties).toContainEqual({
      path: "/field_5000",
      types: ["number"],
      observations: 1,
    });
  });
});
