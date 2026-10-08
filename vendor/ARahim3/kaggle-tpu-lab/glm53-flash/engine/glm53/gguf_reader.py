"""Minimal GGUF reader for the Unsloth GLM-5.3-Flash mirrors: header/tensor-map parsing plus positional reads that
work either on whole .gguf files or on the 4 GB byte-range pieces produced by kaggle/glm_mirror_gguf.py
(manifest.json next to the pieces). Reads use pread (no mmap → no page-cache pressure on the TPU host cgroup).

Tensor naming (llama.cpp glm5next): blk.{L}.ffn_{gate,up,down}_exps.weight with GGUF dims [ne0=in, ne1=out, ne2=E]
i.e. row-major numpy shape (E, out, in); quant blocks run along `in`.
"""
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

GGML_SIZES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
              14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
              22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2)}
GGML_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              29: "IQ1_M", 30: "BF16"}


class PiecewiseFile:
    """A logical file backed by one real file or by ordered byte-range pieces."""

    def __init__(self, pieces):          # pieces: list of (path, start, length) covering [0, size)
        self.pieces = sorted(pieces, key=lambda p: p[1])
        self.size = self.pieces[-1][1] + self.pieces[-1][2]
        self._fds = {}

    @classmethod
    def single(cls, path):
        return cls([(path, 0, os.path.getsize(path))])

    def _fd(self, path):
        if path not in self._fds:
            self._fds[path] = os.open(path, os.O_RDONLY)
        return self._fds[path]

    def pread(self, offset, length):
        out = bytearray(length)
        mv = memoryview(out)
        done = 0
        for path, start, plen in self.pieces:
            if start + plen <= offset + done or start > offset + length:
                continue
            lo = max(offset + done, start)
            hi = min(offset + length, start + plen)
            if hi <= lo:
                continue
            got = os.pread(self._fd(path), hi - lo, lo - start)
            mv[lo - offset:hi - offset] = got
            done = hi - offset
            if done >= length:
                break
        assert done == length, f"short read {done}/{length} at {offset}"
        return out

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def open_mirror(dirs):
    """dirs: dataset mount dirs containing manifest.json + pieces (or raw .gguf files). Returns {shard_name: PiecewiseFile}."""
    files = {}
    for d in dirs:
        man = os.path.join(d, "manifest.json")
        if os.path.exists(man):
            m = json.load(open(man))
            for name, info in m["files"].items():
                files[name] = PiecewiseFile([(os.path.join(d, p["name"]), p["start"], p["length"]) for p in info["pieces"]])
        for fn in os.listdir(d):
            if fn.endswith(".gguf"):
                files[fn] = PiecewiseFile.single(os.path.join(d, fn))
    return files


class GGUFShard:
    def __init__(self, pf: PiecewiseFile):
        self.pf = pf
        self.kv, self.tensors, self.data0 = self._parse()

    def _parse(self):
        buf = self.pf.pread(0, min(self.pf.size, 64 << 20))
        pos = [0]

        def rd(fmt):
            v = struct.unpack_from(fmt, buf, pos[0])[0]
            pos[0] += struct.calcsize(fmt)
            return v

        def rstr():
            n = rd("<Q")
            s = bytes(buf[pos[0]:pos[0] + n]).decode(errors="replace")
            pos[0] += n
            return s

        def rval(t):
            if t in (0, 1): return rd("<B" if t == 0 else "<b")
            if t in (2, 3): return rd("<H" if t == 2 else "<h")
            if t in (4, 5): return rd("<I" if t == 4 else "<i")
            if t == 6: return rd("<f")
            if t == 7: return rd("<B")
            if t == 8: return rstr()
            if t == 9:
                et = rd("<I"); n = rd("<Q"); return [rval(et) for _ in range(n)]
            if t in (10, 11): return rd("<Q" if t == 10 else "<q")
            if t == 12: return rd("<d")
            raise ValueError(t)

        assert bytes(buf[:4]) == b"GGUF", "not a GGUF file"
        pos[0] = 4
        rd("<I"); nt = rd("<Q"); nkv = rd("<Q")
        kv = {}
        for _ in range(nkv):
            k = rstr(); t = rd("<I"); v = rval(t)
            if not (isinstance(v, list) and len(v) > 64):
                kv[k] = v
        align = kv.get("general.alignment", 32)
        tensors = {}
        for _ in range(nt):
            name = rstr(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]; ty = rd("<I"); off = rd("<Q")
            n = 1
            for d in dims:
                n *= d
            bs, tb = GGML_SIZES[ty]
            tensors[name] = {"dims": dims, "type": GGML_NAMES.get(ty, ty), "offset": off, "nbytes": n // bs * tb}
        data0 = (pos[0] + align - 1) // align * align
        return kv, tensors, data0

    def read_tensor(self, name, offset=0, length=None):
        t = self.tensors[name]
        length = t["nbytes"] - offset if length is None else length
        return self.pf.pread(self.data0 + t["offset"] + offset, length)


class GGUFModel:
    """All shards of a split GGUF merged into one tensor map."""

    def __init__(self, files):
        self.shards = {n: GGUFShard(pf) for n, pf in sorted(files.items())}
        self.where = {}
        self.kv = {}
        for n, s in self.shards.items():
            self.kv.update(s.kv)
            for t in s.tensors:
                self.where[t] = n

    def info(self, name):
        return self.shards[self.where[name]].tensors[name]

    def read(self, name, offset=0, length=None):
        return self.shards[self.where[name]].read_tensor(name, offset, length)

    def expert_bytes(self, name):
        """bytes of one expert (dims [in, out, E]) = out * in/bs * tb."""
        t = self.info(name)
        return t["nbytes"] // t["dims"][2]

    def read_expert(self, name, e):
        return self.read(name, e * self.expert_bytes(name), self.expert_bytes(name))

    def read_experts(self, name, es, threads=8):
        with ThreadPoolExecutor(threads) as pool:
            return list(pool.map(lambda e: self.read_expert(name, e), es))
