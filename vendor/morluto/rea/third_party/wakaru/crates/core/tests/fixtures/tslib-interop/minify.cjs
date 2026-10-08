const fs = require("node:fs");
const path = require("node:path");
const { minify } = require("terser");
(async () => {
  const source = fs.readFileSync(path.join(__dirname, "generated/star.js"), "utf8");
  for (const mangle of [false, true]) {
    const result = await minify(source, { module: true, compress: { defaults: true, unused: false }, mangle, format: { comments: false } });
    fs.writeFileSync(path.join(__dirname, `generated/star-${mangle ? "mangled" : "compressed"}.js`), result.code + "\n");
  }
})();
