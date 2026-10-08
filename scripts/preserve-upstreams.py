#!/usr/bin/env python3
"""Import pinned source snapshots as self-contained, squashed Git subtrees."""
import argparse
import configparser
import datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'upstreams.json'


def git(*args, env=None, input=None):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=ROOT,
                                   env=env, input=input).decode().strip()


def fetch(url, revision):
    subprocess.run(['git', '-c', 'gc.auto=0', '-c', 'submodule.recurse=false',
                    'fetch', '--no-tags', '--depth=1', url, revision], cwd=ROOT, check=True)
    return git('rev-parse', 'FETCH_HEAD')


def flatten(url, commit, records, trail=()):
    """Replace gitlinks with their pinned source trees, including nested links."""
    identity = (url, commit)
    if identity in trail:
        raise RuntimeError(f'Recursive submodule cycle: {identity}')
    links = []
    for entry in git('ls-tree', '-r', commit).splitlines():
        metadata, path = entry.split('\t', 1)
        mode, kind, oid = metadata.split()
        if mode == '160000':
            links.append((path, oid))
    if not links:
        return git('rev-parse', f'{commit}^{{tree}}')
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
            child_tree = flatten(child_url, child_commit, child_records, (*trail, identity))
            records.append({'path': path, 'url': child_url, 'commit': oid,
                            'tree': child_tree, 'submodules': child_records})
            git('update-index', '--force-remove', '--', path, env=env)
            git('read-tree', f'--prefix={path}/', child_tree, env=env)
        return git('write-tree', env=env)


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
        tree = flatten(entry['url'], commit, submodules)
        # A source-complete commit makes git subtree preserve nested submodules too.
        message = f"Preserved {entry['url']} at {commit}; flattened pinned submodule trees\n"
        snapshot = git('commit-tree', tree, input=message.encode())
        subprocess.run(['git', 'subtree', 'add', f"--prefix={entry['path']}", '--squash',
                        '-m', f"vendor: preserve {entry['name']} at {commit[:12]}", snapshot],
                       cwd=ROOT, check=True)
        saved = {'commit': commit, 'tree': tree, 'snapshot_commit': snapshot,
                 'submodules': submodules,
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
