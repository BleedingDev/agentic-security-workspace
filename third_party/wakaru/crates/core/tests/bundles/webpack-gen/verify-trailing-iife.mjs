// Optional execution oracle for the checked-in production fixture.
// WAKARU=/absolute/path/to/wakaru node verify-trailing-iife.mjs
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const binary = process.env.WAKARU;
assert.ok(binary, 'Set WAKARU to the CLI binary built from the checkout under test.');
const fixtureDir = dirname(fileURLToPath(import.meta.url));
const original = join(fixtureDir, 'dist/wp5-trailing-iife-min/bundle.js');
const scratch = mkdtempSync(join(tmpdir(), 'wakaru-trailing-iife-'));
const worker = `
  globalThis.document = { body: { innerHTML: '' } };
  globalThis.failLoad = process.argv[2] === 'reject';
  const events = [];
  process.on('unhandledRejection', error => events.push('unhandled: ' + error.message));
  await import(process.argv[1]).then(
    () => events.push('loaded'),
    error => events.push('module rejected: ' + error.message),
  );
  await new Promise(resolve => setTimeout(resolve, 20));
  console.log(JSON.stringify({ events, html: document.body.innerHTML }));
`;
function execute(filename, mode) {
  return JSON.parse(execFileSync(process.execPath, [
    '--input-type=module', '-e', worker, pathToFileURL(filename).href, mode,
  ], { encoding: 'utf8' }));
}
try {
  const recoveredDir = join(scratch, 'recovered');
  execFileSync(binary, [original, '--unpack=auto', '-o', recoveredDir]);
  writeFileSync(join(recoveredDir, 'package.json'), '{"type":"module"}\n');
  for (const mode of ['resolve', 'reject']) {
    const expected = execute(original, mode);
    assert.deepEqual(expected, mode === 'resolve'
      ? { events: ['loaded'], html: '<p>value=43</p>' }
      : { events: ['loaded', 'unhandled: load failed'], html: '' });
    const actual = execute(join(recoveredDir, 'entry.js'), mode);
    assert.deepEqual(actual, expected, `${mode}: recovered module changed async startup behavior`);
    console.log(`${mode}: ${JSON.stringify(actual)}`);
  }
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
