#!/usr/bin/env python3
"""Import pinned source snapshots as self-contained, squashed Git subtrees."""
import argparse
import configparser
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'upstreams.json'
ATTRIBUTE_MARKER = b'# Preserved snapshot: retain raw bytes and embedded LFS payloads.\n** -text -filter\n'


def git(*args, env=None, input=None):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=ROOT,
                                   env=env, input=input).decode(errors='surrogateescape').strip()


def materialize_lfs(url, tree, records):
    result = subprocess.run(['git', 'grep', '-z', '-l', '-F',
                             'version https://git-lfs.github.com/spec/v1', tree],
                            cwd=ROOT, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode(errors='replace'))
    pointers = []
    for match in result.stdout.split(b'\0'):
        if not match:
            continue
        spec = match.decode(errors='surrogateescape')
        raw = git('show', spec)
        if not raw.startswith('version https://git-lfs.github.com/spec/v1\n'):
            continue
        lines = raw.splitlines()
        oid = next(x.removeprefix('oid sha256:') for x in lines if x.startswith('oid sha256:'))
        size = int(next(x.removeprefix('size ') for x in lines if x.startswith('size ')))
        if size >= 100 * 1024 * 1024:
            raise RuntimeError(f'LFS object needs independent large-object storage: {url} {oid} {size}')
        pointers.append({'path': spec.split(':', 1)[1], 'sha256': oid, 'bytes': size})
    if not pointers:
        return tree
    print(f'Materializing {len(pointers)} LFS pointers from {url}', flush=True)
    with tempfile.TemporaryDirectory(prefix='vendor-lfs-', dir=git('rev-parse', '--git-dir')) as tmp:
        tmp = Path(tmp).resolve()
        env = dict(os.environ, GIT_INDEX_FILE=str(tmp / 'index'))
        git('read-tree', tree, env=env)
        objects = {x['sha256']: x['bytes'] for x in pointers}
        actions = {}
        items = list(objects.items())
        for start in range(0, len(items), 100):
            body = {'operation': 'download', 'transfers': ['basic'],
                    'objects': [{'oid': oid, 'size': size} for oid, size in items[start:start+100]]}
            response = subprocess.check_output(['curl', '-fsSL', '--max-time', '60',
                '-H', 'Accept: application/vnd.git-lfs+json', '-H', 'Content-Type: application/vnd.git-lfs+json',
                '--data-binary', json.dumps(body), url.rstrip('/') + '/info/lfs/objects/batch'])
            for obj in json.loads(response)['objects']:
                if 'error' in obj:
                    raise RuntimeError(f"LFS origin object unavailable: {url} {obj['oid']} {obj['error']}")
                actions[obj['oid']] = obj['actions']['download']

        def download(item):
            oid, size = item
            path = tmp / oid
            action = actions[oid]
            command = ['curl', '-fsSL', '--retry', '2', '--max-time', '300', '-o', str(path)]
            for key, value in action.get('header', {}).items():
                command.extend(['-H', f'{key}: {value}'])
            command.append(action['href'])
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if path.stat().st_size != size or digest.hexdigest() != oid:
                raise RuntimeError(f'LFS integrity mismatch: {oid}')
            return oid, git('hash-object', '-w', '--no-filters', str(path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            blobs = dict(pool.map(download, items))
        listing = subprocess.check_output(['git', 'ls-tree', '-rz', tree], cwd=ROOT)
        modes = {}
        attributes = []
        for entry in listing.split(b'\0'):
            if not entry:
                continue
            metadata, path = entry.split(b'\t', 1)
            name = path.decode(errors='surrogateescape')
            modes[name] = metadata.split()[0].decode()
            if name.rsplit('/', 1)[-1] == '.gitattributes':
                attributes.append(name)
        for pointer in pointers:
            path = pointer['path']
            git('update-index', '--add', '--cacheinfo', modes[path], blobs[pointer['sha256']], path, env=env)
        # The archived payloads are normal Git blobs, not filters into an external LFS server.
        for path in attributes:
            original = subprocess.check_output(['git', 'show', f'{tree}:{path}'], cwd=ROOT)
            changed = original.replace(b'filter=lfs', b'-filter').replace(b'diff=lfs', b'-diff').replace(b'merge=lfs', b'-merge')
            if changed != original:
                backup = path + '.upstream'
                if backup in modes:
                    raise RuntimeError(f'Attribute backup collision: {backup}')
                oid = git('hash-object', '-w', '--stdin', input=original)
                git('update-index', '--add', '--cacheinfo', '100644', oid, backup, env=env)
                oid = git('hash-object', '-w', '--stdin', input=changed)
                git('update-index', '--cacheinfo', modes[path], oid, path, env=env)
        records.extend(pointers)
        return git('write-tree', env=env)


def preserve_attributes(tree):
    """Keep exact fixture bytes even when upstream enables text normalization."""
    listing = subprocess.check_output(['git', 'ls-tree', '-rz', tree], cwd=ROOT)
    entries = {}
    for item in listing.split(b'\0'):
        if item:
            metadata, path = item.split(b'\t', 1)
            entries[path.decode(errors='surrogateescape')] = metadata.split()[0].decode()
    paths = [p for p in entries if p.rsplit('/', 1)[-1] == '.gitattributes']
    if '.gitattributes' not in paths:
        paths.append('.gitattributes')
    with tempfile.TemporaryDirectory(prefix='vendor-attributes-', dir=git('rev-parse', '--git-dir')) as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp).resolve() / 'index'))
        git('read-tree', tree, env=env)
        for path in paths:
            original = subprocess.check_output(['git', 'show', f'{tree}:{path}'], cwd=ROOT) if path in entries else b''
            if original.endswith(ATTRIBUTE_MARKER):
                continue
            if original and path + '.upstream' not in entries:
                oid = git('hash-object', '-w', '--stdin', input=original)
                git('update-index', '--add', '--cacheinfo', '100644', oid, path + '.upstream', env=env)
            changed = original.rstrip(b'\n') + b'\n' + ATTRIBUTE_MARKER
            oid = git('hash-object', '-w', '--stdin', input=changed)
            git('update-index', '--add', '--cacheinfo', '100644', oid, path, env=env)
        return git('write-tree', env=env)


def fetch(url, revision):
    subprocess.run(['git', '-c', 'gc.auto=0', '-c', 'submodule.recurse=false',
                    'fetch', '--no-tags', '--depth=1', url, revision], cwd=ROOT, check=True)
    return git('rev-parse', 'FETCH_HEAD')


def flatten(url, commit, records, lfs, trail=()):
    """Replace gitlinks with their pinned source trees, including nested links."""
    identity = (url, commit)
    if identity in trail:
        raise RuntimeError(f'Recursive submodule cycle: {identity}')
    links = []
    listing = subprocess.check_output(['git', 'ls-tree', '-rz', commit], cwd=ROOT)
    for raw in listing.split(b'\0'):
        if not raw:
            continue
        entry = raw.decode(errors='surrogateescape')
        metadata, path = entry.split('\t', 1)
        mode, kind, oid = metadata.split()
        if mode == '160000':
            links.append((path, oid))
    if not links:
        return preserve_attributes(materialize_lfs(url, git('rev-parse', f'{commit}^{{tree}}'), lfs))
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(git('show', f'{commit}:.gitmodules'))
    urls = {config[s]['path']: config[s]['url'] for s in config.sections()}
    with tempfile.TemporaryDirectory(prefix='vendor-index-', dir=git('rev-parse', '--git-dir')) as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp).resolve() / 'index'))
        git('read-tree', commit, env=env)
        for path, oid in links:
            child_url = urls[path]
            if child_url.startswith('.'):
                child_url = urljoin(url.rstrip('/') + '/', child_url)
            if not child_url.startswith('https://'):
                raise RuntimeError(f'Unsupported submodule URL, review it explicitly: {child_url}')
            child_commit = fetch(child_url, oid)
            if child_commit != oid:
                raise RuntimeError(f'Submodule revision changed: {path}')
            child_records = []
            child_lfs = []
            child_tree = flatten(child_url, child_commit, child_records, child_lfs, (*trail, identity))
            records.append({'path': path, 'url': child_url, 'commit': oid,
                            'tree': child_tree, 'submodules': child_records, 'lfs': child_lfs})
            git('update-index', '--force-remove', '--', path, env=env)
            git('read-tree', f'--prefix={path}/', child_tree, env=env)
        return preserve_attributes(materialize_lfs(url, git('write-tree', env=env), lfs))


def verify(entry):
    actual = git('rev-parse', f"HEAD:{entry['path']}")
    if actual != entry['tree']:
        raise RuntimeError(f"Snapshot differs from recorded tree: {entry['path']}")
    if any(line.startswith('160000 ') for line in git('ls-tree', '-r', actual).splitlines()):
        raise RuntimeError(f"External gitlink remains: {entry['path']}")
    print(f"Verified {entry['path']} {entry['commit'][:12]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['import', 'verify'])
    parser.add_argument('--name', help='Import only this manifest entry')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.action == 'verify':
        for entry in manifest['sources']:
            verify(entry)
        return
    if git('status', '--porcelain'):
        raise RuntimeError('Import requires a clean worktree; preserve unrelated changes first')
    state_file = Path(git('rev-parse', '--git-dir')) / 'preserved-upstreams.json'
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        state = {}
    for entry in manifest['sources']:
        if args.name and entry['name'] != args.name:
            continue
        if entry['name'] in state:
            entry.update(state[entry['name']])
            verify(entry)
            continue
        if (ROOT / entry['path']).exists():
            raise RuntimeError(f"Unrecorded existing destination: {entry['path']}")
        revision = entry.get('commit') or f"refs/heads/{entry['branch']}"
        commit = fetch(entry['url'], revision)
        submodules = []
        lfs = []
        tree = flatten(entry['url'], commit, submodules, lfs)
        # A source-complete commit makes git subtree preserve nested submodules too.
        message = f"Preserved {entry['url']} at {commit}; flattened pinned submodule trees\n"
        snapshot = git('commit-tree', tree, input=message.encode())
        subprocess.run(['git', 'subtree', 'add', f"--prefix={entry['path']}", '--squash',
                        '-m', f"vendor: preserve {entry['name']} at {commit[:12]}", snapshot],
                       cwd=ROOT, check=True)
        saved = {'commit': commit, 'tree': tree, 'snapshot_commit': snapshot,
                 'submodules': submodules,
                 'lfs': lfs,
                 'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        entry.update(saved)
        state[entry['name']] = saved
        state_file.write_text(json.dumps(state, indent=2) + '\n')
        verify(entry)
    # Record provenance after subtree commits so each import starts clean.
    for entry in manifest['sources']:
        if entry['name'] in state:
            entry.update(state[entry['name']])
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
