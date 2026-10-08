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
            command = ['curl', '-fsSL', '--proto', '=https', '--proto-redir', '=https',
                       '--retry', '2', '--max-time', '300', '-o', str(path)]
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


def verify_payloads(tree, entry, checked):
    for payload in entry.get('lfs', []):
        blob = git('rev-parse', tree + ':' + payload['path'])
        if blob in checked:
            digest, size = checked[blob]
        else:
            data = subprocess.check_output(['git', 'cat-file', 'blob', blob], cwd=ROOT)
            digest, size = hashlib.sha256(data).hexdigest(), len(data)
            checked[blob] = digest, size
        if digest != payload['sha256'] or size != payload['bytes']:
            raise RuntimeError(f"Archived LFS payload differs: {payload['path']}")
    for child in entry.get('submodules', []):
        child_tree = git('rev-parse', tree + ':' + child['path'])
        if child_tree != child['tree']:
            raise RuntimeError(f"Archived submodule tree differs: {child['path']}")
        verify_payloads(child_tree, child, checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['import', 'update', 'verify'])
    parser.add_argument('--name', help='Import only this manifest entry')
    parser.add_argument('--revision', help='Exact source revision to import or update')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.action == 'update' and not args.name:
        raise RuntimeError('Update requires one explicit --name')
    if args.name and not any(e['name'] == args.name for e in manifest['sources']):
        raise RuntimeError(f'Unknown source: {args.name}')
    for entry in manifest['sources']:
        path = Path(entry['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'vendor':
            raise RuntimeError(f'Unsafe archive destination: {path}')
    if args.action == 'verify':
        checked = {}
        for entry in manifest['sources']:
            verify(entry)
            verify_payloads(entry['tree'], entry, checked)
        for resource in manifest.get('external_resources', []):
            if resource.get('archive_status') != 'archived':
                continue
            data = subprocess.check_output(['git', 'show', 'HEAD:' + resource['path']], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != resource['sha256']:
                raise RuntimeError(f"Resource checksum differs: {resource['path']}")
        for model in manifest.get('models', []):
            for file in model['files']:
                if not file.get('archived_sha256'):
                    continue
                name = file['path'] + ('.upstream' if file['path'] == '.gitattributes' else '')
                path = model['metadata_path'] + '/' + name
                data = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)
                if len(data) != file['bytes'] or hashlib.sha256(data).hexdigest() != file['archived_sha256']:
                    raise RuntimeError(f'Model metadata differs: {path}')
        print(f'Verified {len(checked)} unique embedded LFS payloads; model payload status is separate')
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
        if args.action == 'import' and entry.get('tree'):
            verify(entry)
            continue
        if args.action == 'import' and entry['name'] in state:
            entry.update(state[entry['name']])
            verify(entry)
            continue
        if args.action == 'import' and (ROOT / entry['path']).exists():
            raise RuntimeError(f"Unrecorded existing destination: {entry['path']}")
        revision = args.revision or (f"refs/heads/{entry['branch']}" if args.action == 'update'
                                    else entry.get('commit') or f"refs/heads/{entry['branch']}")
        commit = fetch(entry['url'], revision)
        submodules = []
        lfs = []
        tree = flatten(entry['url'], commit, submodules, lfs)
        # A source-complete commit makes git subtree preserve nested submodules too.
        message = f"Preserved {entry['url']} at {commit}; flattened pinned submodule trees\n"
        snapshot = git('commit-tree', tree, input=message.encode())
        operation = 'merge' if args.action == 'update' else 'add'
        subprocess.run(['git', '-c', 'gc.auto=0', 'subtree', operation, f"--prefix={entry['path']}", '--squash',
                        '-m', f"vendor: preserve {entry['name']} at {commit[:12]}", snapshot],
                       cwd=ROOT, check=True)
        # Squashed subtree history otherwise keeps the split commit only as text.
        # Retain its object as an ancestor so a fresh clone can update offline.
        subprocess.run(['git', '-c', 'gc.auto=0', 'merge', '-s', 'ours', '--no-ff',
                        '--allow-unrelated-histories', '-m',
                        f"vendor: retain snapshot identity for {entry['name']}", snapshot],
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
        if entry['name'] in state and (not entry.get('tree') or args.action == 'update' and entry['name'] == args.name):
            entry.update(state[entry['name']])
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
