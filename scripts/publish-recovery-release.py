#!/usr/bin/env python3
"""Publish a pinned artifact backup using an operation-owned temporary directory."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 2 * 1024**3


def run(*args, cwd=ROOT, **kwargs):
    return subprocess.run(args, cwd=cwd, check=True, **kwargs)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(entry, destination):
    name = entry['name']
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Unsafe asset name')
    path = destination / name
    run('curl', '-fsSL', '--proto', '=https', '--proto-redir', '=https',
        '--retry', '3', '--max-time', '900', '-o', str(path), entry['url'])
    digest = sha256(path)
    size = path.stat().st_size
    if size >= LIMIT or entry.get('bytes') is not None and size != entry['bytes']:
        raise ValueError('Asset size differs: ' + name)
    if entry.get('sha256') and digest != entry['sha256']:
        raise ValueError('Asset checksum differs: ' + name)
    return {**entry, 'bytes': size, 'sha256': digest}


def recursive_source(entry, destination):
    spec = importlib.util.spec_from_file_location('preserve', ROOT / 'scripts/preserve-upstreams.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='source-', dir=destination) as directory:
        module.ROOT = Path(directory)
        run('git', 'init', '-q', directory)
        # Borrow existing source objects read-only; newly fetched objects are temporary.
        objects = subprocess.check_output(['git', 'rev-parse', '--git-path', 'objects'], cwd=ROOT).decode().strip()
        (module.ROOT / '.git/objects/info/alternates').write_text(str((ROOT / objects).resolve()) + '\n')
        commit = module.fetch('https://github.com/' + entry['project'] + '.git', entry['commit'])
        submodules, lfs = [], []
        tree = module.flatten('https://github.com/' + entry['project'] + '.git', commit, submodules, lfs)
        module.verify_payloads(tree, {'submodules': submodules, 'lfs': lfs}, {})
        path = destination / entry['name']
        run('git', 'archive', '--format=tar.gz', '--prefix=source/', '-o', str(path), tree, cwd=module.ROOT)
        if path.stat().st_size >= LIMIT:
            raise ValueError('Source archive exceeds asset limit')
        return {**entry, 'kind': 'recursive-source', 'tree': tree, 'submodules': submodules,
                'lfs': lfs, 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def rea_dependencies(version, destination):
    with tempfile.TemporaryDirectory(prefix='rea-', dir=destination) as directory:
        directory = Path(directory)
        package = {'name': 'rea-recovery', 'version': '1.0.0', 'private': True,
                   'dependencies': {'rea-agents': version}}
        (directory / 'package.json').write_text(json.dumps(package, indent=2) + '\n')
        cache = directory / 'npm-cache'
        run('npm', 'install', '--ignore-scripts', '--no-audit', '--no-fund',
            '--registry=https://registry.npmjs.org', '--cache=' + str(cache), cwd=directory)
        lock = json.loads((directory / 'package-lock.json').read_text())
        packages = []
        for name, item in lock['packages'].items():
            if not name or not item.get('resolved'):
                continue
            url, integrity = item['resolved'], item.get('integrity', '')
            if not url.startswith('https://registry.npmjs.org/') or not integrity.startswith('sha512-'):
                raise ValueError('Unexpected npm origin/integrity: ' + name)
            packages.append({'path': name, 'version': item['version'], 'url': url,
                             'integrity': integrity, 'license': item.get('license')})
        # Prove that the cache resolves this lockfile without registry access.
        with tempfile.TemporaryDirectory(prefix='offline-', dir=destination) as offline:
            for name in ('package.json', 'package-lock.json'):
                Path(offline, name).write_bytes((directory / name).read_bytes())
            run('npm', 'ci', '--offline', '--ignore-scripts', '--no-audit', '--no-fund',
                '--cache=' + str(cache), cwd=offline)
        (directory / 'RECOVERY.txt').write_text(
            'REA npm dependency snapshot. Restore with npm ci --offline --ignore-scripts '
            '--cache ./npm-cache. Generated on ' + os.uname().sysname + ' ' + os.uname().machine +
            '. Platform optional packages may differ. Native postinstall engines, external '
            'analysis tools and Node itself are not included. Packages retain their licenses.\n')
        # npm logs are operational data, not recovery artifacts.
        path = destination / ('rea-' + version + '-npm-recovery.tar.gz')
        with tarfile.open(path, 'w:gz') as archive:
            for name in ('package.json', 'package-lock.json', 'RECOVERY.txt', 'node_modules'):
                archive.add(directory / name, arcname='rea-recovery/' + name)
            archive.add(cache / '_cacache', arcname='rea-recovery/npm-cache/_cacache')
        if path.stat().st_size >= LIMIT:
            raise ValueError('REA archive exceeds asset limit')
        return {'name': path.name, 'kind': 'npm-dependency-snapshot', 'version': version,
                'platform': os.uname().sysname + '/' + os.uname().machine,
                'offline_dependency_resolution': 'passed-with-install-scripts-disabled',
                'packages': packages, 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--repo', default='BleedingDev/agentic-security-workspace')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag):
        raise ValueError('Unsafe release tag')
    owned = os.environ.get('OWNED_TEMP_DIR')
    if not owned:
        raise RuntimeError('Run through owned-temp-dir --run recovery-release -- ...')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise RuntimeError('Commit the release plan and documentation first')
    destination = Path(owned) / 'release'
    destination.mkdir()  # This operation exclusively owns this new directory.
    plan = json.loads((ROOT / 'release-artifacts.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    bundle = destination / 'agentic-security-workspace.bundle'
    run('git', 'bundle', 'create', str(bundle), 'main')
    run('git', 'bundle', 'verify', str(bundle))
    if bundle.stat().st_size >= LIMIT:
        raise ValueError('Git bundle exceeds release asset limit')
    records = [{'name': bundle.name, 'kind': 'git-bundle', 'commit': commit,
                'bytes': bundle.stat().st_size, 'sha256': sha256(bundle)}]
    entries = plan['assets']
    ordinary = [e for e in entries if e['kind'] != 'recursive-source']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(pool.map(lambda entry: download(entry, destination), ordinary))
    for entry in entries:
        if entry['kind'] == 'recursive-source':
            print('Preserving recursive release source:', entry['project'], flush=True)
            records.append(recursive_source(entry, destination))
    records.append(rea_dependencies(plan['rea_npm_version'], destination))
    manifest = {**plan, 'workspace_commit': commit, 'assets': records,
                'total_bytes': sum(e['bytes'] for e in records)}
    (destination / 'recovery-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    checksums = destination / 'SHA256SUMS'
    checksums.write_text(''.join(sha256(path) + '  ' + path.name + '\n'
                               for path in sorted(destination.iterdir()) if path.is_file() and path != checksums))
    notes = destination / 'release-notes.txt'
    notes.write_text('Recovery artifact backup without model weights\n\n'
        + str(len(records)) + ' archived artifacts, ' + str(round(manifest['total_bytes'] / 1e9, 3))
        + ' GB. Workspace commit: ' + commit + '.\n\n'
        + 'Verify with: shasum -a 256 -c SHA256SUMS\n'
        + 'Restore sources with: git clone agentic-security-workspace.bundle workspace\n\n'
        + 'Original licenses apply. SHA256SUMS and recovery-manifest.json describe exact '
        + 'copies and upstream provenance. Recursive release sources retain submodules '
        + 'and embedded LFS contents. Installed tool versions are unchanged.\n\n'
        + 'Coverage gaps:\n' + ''.join('- ' + gap + '\n' for gap in plan['gaps']))
    run('gh', 'release', 'create', args.tag, '--repo', args.repo, '--target', commit,
        '--draft', '--title', 'Recovery backup without model weights', '--notes-file', str(notes))
    for record in records + [{'name': 'recovery-manifest.json'}, {'name': 'SHA256SUMS'}]:
        print('Uploading:', record['name'], flush=True)
        run('gh', 'release', 'upload', args.tag, str(destination / record['name']), '--repo', args.repo)
    # The by-tag endpoint only returns published releases, not drafts.
    releases = json.loads(subprocess.check_output(['gh', 'api',
        'repos/' + args.repo + '/releases?per_page=100']))
    remote = next(release for release in releases if release['tag_name'] == args.tag)
    actual = {entry['name']: entry for entry in remote['assets']}
    for path in destination.iterdir():
        if not path.is_file() or path == notes:
            continue
        entry = actual[path.name]
        if entry['size'] != path.stat().st_size or entry.get('digest') != 'sha256:' + sha256(path):
            raise ValueError('Uploaded asset differs: ' + path.name)
    run('gh', 'release', 'edit', args.tag, '--repo', args.repo, '--draft=false')
    print(json.dumps({'url': remote['html_url'], 'bytes': manifest['total_bytes'],
                      'artifacts': len(records), 'verified': True}), flush=True)


if __name__ == '__main__':
    main()
