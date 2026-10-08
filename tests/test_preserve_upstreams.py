import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'preserve_upstreams', Path(__file__).resolve().parents[1] / 'scripts/preserve-upstreams.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='preservation-test-',
                                               dir=os.environ.get('OWNED_TEMP_DIR'))
        self.old_root = MODULE.ROOT
        MODULE.ROOT = Path(self.temp.name)
        subprocess.run(['git', 'init', '-q', self.temp.name], check=True)
        MODULE.git('config', 'user.name', 'Fixture')
        MODULE.git('config', 'user.email', 'fixture@example.invalid')

    def tearDown(self):
        MODULE.ROOT = self.old_root
        self.temp.cleanup()

    def blob(self, data):
        return MODULE.git('hash-object', '-w', '--stdin', input=data)

    def tree(self, entries):
        return MODULE.git('mktree', '-z', input=b''.join(
            f'{mode} {kind} {oid}\t'.encode() + path + b'\0'
            for mode, kind, oid, path in entries))

    def test_nested_submodule_survives_without_origin(self):
        child_tree = self.tree([('100644', 'blob', self.blob(b'nested source\n'), b'LICENSE')])
        child = MODULE.git('commit-tree', child_tree, input=b'child\n')
        config = b'[submodule "nested"]\n path = nested\n url = ../child.git\n'
        parent_tree = self.tree([('100644', 'blob', self.blob(config), b'.gitmodules'),
                                 ('160000', 'commit', child, b'nested')])
        parent = MODULE.git('commit-tree', parent_tree, input=b'parent\n')
        old_fetch = MODULE.fetch
        MODULE.fetch = lambda url, revision: revision
        try:
            records = []
            preserved = MODULE.flatten('https://example.invalid/owner/parent.git', parent, records, [])
        finally:
            MODULE.fetch = old_fetch
        self.assertNotIn('160000 ', MODULE.git('ls-tree', '-r', preserved))
        self.assertEqual(MODULE.git('show', preserved + ':nested/LICENSE'), 'nested source')
        self.assertEqual(records[0]['url'], 'https://example.invalid/owner/child.git')
        MODULE.verify_payloads(preserved, {'submodules': records}, {})

    def test_text_attributes_preserve_original_crlf_bytes(self):
        original = b'* text=auto\n'
        content = b'first\r\nsecond\r\n'
        tree = self.tree([('100644', 'blob', self.blob(original), b'.gitattributes'),
                          ('100644', 'blob', self.blob(content), b'fixture.txt')])
        preserved = MODULE.preserve_attributes(tree)
        self.assertEqual(MODULE.git('rev-parse', preserved + ':fixture.txt'), self.blob(content))
        self.assertEqual(subprocess.check_output(['git', 'show', preserved + ':.gitattributes.upstream'],
                                                cwd=MODULE.ROOT), original)
        MODULE.git('read-tree', preserved)
        MODULE.git('checkout-index', '--all')
        self.assertEqual((MODULE.ROOT / 'fixture.txt').read_bytes(), content)
        MODULE.git('update-index', '--refresh')
        self.assertEqual(MODULE.git('diff-files', '--name-only'), '')
        self.assertEqual(MODULE.preserve_attributes(preserved), preserved)

    def test_tab_and_unicode_paths_remain_byte_exact(self):
        path = 'část\tfile.txt'.encode()
        tree = self.tree([('100644', 'blob', self.blob(b'exact\n'), path)])
        preserved = MODULE.preserve_attributes(tree)
        self.assertEqual(MODULE.git('show', preserved + ':' + path.decode()), 'exact')

    def test_wrong_lfs_checksum_is_rejected(self):
        tree = self.tree([('100644', 'blob', self.blob(b'payload'), b'asset')])
        with self.assertRaisesRegex(RuntimeError, 'LFS payload differs'):
            MODULE.verify_payloads(tree, {'lfs': [{'path': 'asset', 'sha256': '0' * 64, 'bytes': 7}]}, {})


if __name__ == '__main__':
    unittest.main()
