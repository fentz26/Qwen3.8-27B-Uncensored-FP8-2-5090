#!/usr/bin/env python3
"""
Unit tests for GPU selection in run.py. No GPU required — nvidia-smi output is
synthetic.

Regression under test: gpu_snapshot() used to sum every GPU on the host, so a
1-GPU profile on a 4-GPU box reported ~4x the real VRAM.

  python3 -m unittest discover -s bench -p 'test_*.py'
  python3 bench/test_gpu_snapshot.py
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "bench_run", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def smi(n, used=20000, util=95, power=400):
    """Synthetic `nvidia-smi --query-gpu=...` CSV for an n-GPU host."""
    return "\n".join(
        f"{i}, NVIDIA GeForce RTX 5090, {used}, {util}, {power}, GPU-uuid-{i}"
        for i in range(n))


class TestParsing(unittest.TestCase):
    def test_parses_rows(self):
        rows = run.parse_nvidia_smi_csv(smi(2))
        self.assertEqual([r["index"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["vram_mib"], 20000.0)
        self.assertEqual(rows[0]["uuid"], "GPU-uuid-0")

    def test_unsupported_sensor_becomes_none_not_zero(self):
        # '[N/A]' must not become 0.0 — that would assert a measurement.
        rows = run.parse_nvidia_smi_csv(
            "0, NVIDIA GeForce RTX 5090, 20000, 95, [N/A], GPU-uuid-0")
        self.assertIsNone(rows[0]["power_w"])
        self.assertEqual(rows[0]["vram_mib"], 20000.0)

    def test_ignores_blank_and_malformed_lines(self):
        self.assertEqual(run.parse_nvidia_smi_csv("\n\nnot,enough\n"), [])


class TestSelection(unittest.TestCase):
    def test_one_selected_gpu_on_two_gpu_host(self):
        """Required case 1: the original bug — 1-GPU profile on a 2-GPU host."""
        rows = run.parse_nvidia_smi_csv(smi(2))
        sel, src = run.resolve_gpu_selection(rows, cuda_visible_devices="0")
        self.assertEqual([g["index"] for g in sel], [0])
        self.assertEqual(src, "CUDA_VISIBLE_DEVICES")
        agg = run.summarize_gpus(sel)
        self.assertEqual(agg["vram_mib"], 20000.0)      # not 40000
        self.assertEqual(agg["power_w"], 400.0)         # not 800
        self.assertEqual(len(agg["per_gpu"]), 1)

    def test_two_selected_gpus_on_four_gpu_host(self):
        """Required case 2."""
        rows = run.parse_nvidia_smi_csv(smi(4))
        sel, _ = run.resolve_gpu_selection(rows, cuda_visible_devices="0,1")
        self.assertEqual([g["index"] for g in sel], [0, 1])
        agg = run.summarize_gpus(sel)
        self.assertEqual(agg["vram_mib"], 40000.0)      # not 80000
        self.assertEqual(agg["gpu_util_pct"], 95.0)     # mean over selection

    def test_cuda_visible_devices_remapping_2_3(self):
        """Required case 3: CUDA_VISIBLE_DEVICES='2,3' on a 4-GPU host.

        nvidia-smi reports PHYSICAL indices and ignores CUDA_VISIBLE_DEVICES,
        so selection must match physical 2 and 3 — not CUDA's remapped 0 and 1.
        """
        rows = run.parse_nvidia_smi_csv(smi(4))
        sel, src = run.resolve_gpu_selection(rows, cuda_visible_devices="2,3")
        self.assertEqual([g["index"] for g in sel], [2, 3])
        self.assertEqual(src, "CUDA_VISIBLE_DEVICES")
        self.assertEqual(run.summarize_gpus(sel)["vram_mib"], 40000.0)

    def test_explicit_overrides_cuda_visible_devices(self):
        rows = run.parse_nvidia_smi_csv(smi(4))
        sel, src = run.resolve_gpu_selection(rows, cuda_visible_devices="0,1,2,3", explicit="3")
        self.assertEqual([g["index"] for g in sel], [3])
        self.assertEqual(src, "explicit")

    def test_unset_falls_back_to_all_host_gpus(self):
        rows = run.parse_nvidia_smi_csv(smi(2))
        sel, src = run.resolve_gpu_selection(rows, cuda_visible_devices=None)
        self.assertEqual([g["index"] for g in sel], [0, 1])
        self.assertEqual(src, "all-host-gpus")

    def test_empty_string_means_no_visible_gpus(self):
        # Distinct from unset: CUDA_VISIBLE_DEVICES="" hides every GPU.
        rows = run.parse_nvidia_smi_csv(smi(2))
        sel, src = run.resolve_gpu_selection(rows, cuda_visible_devices="")
        self.assertEqual(sel, [])
        self.assertEqual(src, "CUDA_VISIBLE_DEVICES(empty)")
        self.assertEqual(run.summarize_gpus(sel), {})

    def test_uuid_selection(self):
        rows = run.parse_nvidia_smi_csv(smi(4))
        sel, _ = run.resolve_gpu_selection(rows, cuda_visible_devices="GPU-uuid-1,GPU-uuid-3")
        self.assertEqual([g["index"] for g in sel], [1, 3])

    def test_invalid_entry_truncates_like_cuda(self):
        # CUDA ignores an invalid device and everything after it.
        rows = run.parse_nvidia_smi_csv(smi(4))
        sel, _ = run.resolve_gpu_selection(rows, cuda_visible_devices="0,9,1")
        self.assertEqual([g["index"] for g in sel], [0])

    def test_duplicates_counted_once(self):
        rows = run.parse_nvidia_smi_csv(smi(2))
        sel, _ = run.resolve_gpu_selection(rows, cuda_visible_devices="0,0")
        self.assertEqual(run.summarize_gpus(sel)["vram_mib"], 20000.0)


class TestAggregate(unittest.TestCase):
    def test_per_gpu_detail_is_recorded(self):
        rows = run.parse_nvidia_smi_csv(smi(4))
        sel, _ = run.resolve_gpu_selection(rows, cuda_visible_devices="2,3")
        per = run.summarize_gpus(sel)["per_gpu"]
        self.assertEqual([g["index"] for g in per], [2, 3])
        self.assertIn("name", per[0])

    def test_missing_sensor_yields_none_not_zero(self):
        rows = run.parse_nvidia_smi_csv(
            "0, NVIDIA GeForce RTX 5090, 20000, 95, [N/A], GPU-uuid-0")
        agg = run.summarize_gpus(rows)
        self.assertIsNone(agg["power_w"])
        self.assertEqual(agg["vram_mib"], 20000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
