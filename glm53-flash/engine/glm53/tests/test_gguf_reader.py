import json, os, numpy as np
from gguf import GGUFWriter, GGMLQuantizationType as T
from glm53 import gguf_reader, iqquant


def _write_model(tmp_path):
    E, out, inn = 4, 8, 512                          # 2 blocks per row
    bb = iqquant.BLOCK_BYTES["IQ2_S"]
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (E, out, inn // 256, bb), dtype=np.uint8)
    raw[..., 0:2] = np.frombuffer(np.full(E * out * 2, 0.01, np.float16).tobytes(), np.uint8).reshape(E, out, 2, 2)
    w = GGUFWriter(str(tmp_path / "m.gguf"), "glm5next")
    w.add_uint32("glm5next.expert_count", E)
    w.add_tensor("blk.3.ffn_gate_exps.weight", raw.reshape(E, out, -1), raw_dtype=T.IQ2_S)
    small = np.arange(64, dtype=np.float32).reshape(8, 8)
    w.add_tensor("blk.0.attn_norm.weight", small)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return raw, small


def test_single_file_and_pieces(tmp_path):
    raw, small = _write_model(tmp_path)
    whole = tmp_path / "m.gguf"
    # cut into 1000-byte pieces + manifest, like the mirror does
    pd = tmp_path / "pieces"; pd.mkdir()
    data = whole.read_bytes(); P = 1000; pieces = []
    for i, s in enumerate(range(0, len(data), P)):
        chunk = data[s:s + P]; (pd / f"m.gguf.p{i:02d}").write_bytes(chunk)
        pieces.append({"name": f"m.gguf.p{i:02d}", "start": s, "length": len(chunk)})
    (pd / "manifest.json").write_text(json.dumps({"files": {"m.gguf": {"size": len(data), "pieces": pieces}}}))
    for files in (gguf_reader.open_mirror([str(tmp_path)]), gguf_reader.open_mirror([str(pd)])):
        m = gguf_reader.GGUFModel(files)
        assert m.kv["glm5next.expert_count"] == 4
        t = m.info("blk.3.ffn_gate_exps.weight")
        assert t["type"] == "IQ2_S" and t["dims"] == [512, 8, 4]
        assert m.expert_bytes("blk.3.ffn_gate_exps.weight") == 8 * 2 * 82
        for e in range(4):
            got = np.frombuffer(m.read_expert("blk.3.ffn_gate_exps.weight", e), np.uint8)
            assert np.array_equal(got, raw[e].reshape(-1))
        got_all = m.read_experts("blk.3.ffn_gate_exps.weight", [3, 1])
        assert np.array_equal(np.frombuffer(got_all[0], np.uint8), raw[3].reshape(-1))
        sm = np.frombuffer(m.read("blk.0.attn_norm.weight"), np.float32).reshape(8, 8)
        assert np.array_equal(sm, small)
        # dequant of a read expert goes through the tested path
        deq = iqquant.dequant_rows(np.frombuffer(m.read_expert("blk.3.ffn_gate_exps.weight", 0), np.uint8), "IQ2_S", 512)
        assert deq.shape == (8, 512)
