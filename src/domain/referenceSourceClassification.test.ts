import { describe, expect, it } from "vitest";

import { classifyReferenceSourcePath } from "./referenceSourceClassification.js";

describe("reference source manifest filenames", () => {
  it.each(["CMakeLists.txt", "cmakelists.txt", "CMAKELISTS.TXT"])(
    "recognizes %s with the existing case-insensitive policy",
    (path) => {
      expect(classifyReferenceSourcePath(path)).toEqual([
        "documentation",
        "manifest",
      ]);
    },
  );

  it("preserves secondary classifications for a nested CMake manifest", () => {
    expect(classifyReferenceSourcePath("src/CMakeLists.txt")).toEqual([
      "documentation",
      "manifest",
      "source",
    ]);
  });

  it.each(["CMakeLists.txt.backup", "CMakeList.txt", "notes.txt"])(
    "does not classify the near-match %s as a manifest",
    (path) => {
      expect(classifyReferenceSourcePath(path)).not.toContain("manifest");
    },
  );

  it.each(["package.json", "PACKAGE.JSON", "Cargo.toml"])(
    "preserves case-insensitive recognition of %s",
    (path) => {
      expect(classifyReferenceSourcePath(path)).toContain("manifest");
    },
  );
});
