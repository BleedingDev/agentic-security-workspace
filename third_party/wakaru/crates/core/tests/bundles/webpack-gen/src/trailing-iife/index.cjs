const load = require('./load.cjs');
const format = require('./format.cjs');
const render = require('./render.cjs');
async function main() {
  const result = await load(42);
  document.body.innerHTML = render(format(result));
}
main();
