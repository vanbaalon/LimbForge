#!/usr/bin/env python3
"""Verify archived prototype outputs; --live also checks frozen local binaries."""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from decimal import Decimal, localcontext
from pathlib import Path

sys.dont_write_bytecode = True
results = Path(__file__).resolve().parents[1] / "results"
baseline = results / "section9_p3_wolfnum_1_3_1_consumer"
spec = importlib.util.spec_from_file_location("comparison_reader", baseline / "comparison_reader.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(candidate, live=False):
    root = Path("/tmp/wolfnum-qsc-section9-" + candidate) if live else None
    folder = root / "prototype-results" if live else results / ("section9_p3_" + {"fourier":"fourier", "small4":"products4", "lu4":"lu4"}[candidate] + "_prototype")
    manifest = json.loads((folder / "gpu_gate_metadata.json").read_text())
    assert manifest["state"] == "terminal" and manifest["exit"] == 0
    if live:
        for filename, digest in manifest["sha256"].items():
            assert sha(root / filename) == digest, filename
    source = baseline / "j3_X2Y_g0.202_nc21.txt"
    reference = baseline / "j3_X2Y_g0.202_nc21_cpu.m"
    expected = reader.read(reference)
    gtol = reader.number(next(line.split()[1] for line in source.read_text().splitlines() if line.startswith("gtol ")))
    records = []
    for step in manifest["steps"]:
        assert step["state"] == "terminal" and step["exit"] == 0
        log = folder / step["log"]
        assert sha(log) == step["log_sha256"], log
        text = log.read_text()
        if not step["log"].startswith("j3_"):
            widths = [int(x) for x in re.findall(r"^(\d+) bits:.*pass$", text, re.M)]
            assert widths == (list(range(64, 1025, 32)) + [374] if candidate == "lu4" else [352, 374, 448]), widths
            records.append({"log": log.name, "checked_precisions": widths, "shader_validation": step["shader_validation"]})
            continue
        output = log.with_suffix(".m")
        if "output_sha256" in step:
            assert sha(output) == step["output_sha256"]
        actual = reader.read(output)
        markers = ["stage 2 (GPU)", "normal matrix (GPU)", "first Cholesky (GPU)", "adjoint check: PASS"]
        combined = step.get("combined_fourier", step.get("combined_candidates", False))
        if candidate == "fourier" or combined:
            markers += ["Fourier Jacobian (GPU)", "exact Fourier:"]
        if candidate == "small4" or combined:
            markers += ["tangent 4x4 Jacobian (GPU)", "tangent products4:"]
        if candidate == "lu4":
            markers += ["descent inverses4 (GPU)", "gluing inverses4 (GPU)", "inverse4 LU:"]
        assert all(marker in text for marker in markers)
        assert "using CPU fallback" not in text and "adjoint check: FAIL" not in text
        assert len(actual[1]) == len(expected[1]) == 3
        with localcontext() as context:
            context.prec = 160
            error = max(abs(x-y) for x,y in zip(actual[0], expected[0]))
            history = max(abs(x-y)/max(abs(x),abs(y),Decimal("1e-300")) for x,y in zip(actual[1],expected[1]))
        assert error <= Decimal("1e-25") and history <= Decimal("1e-10")
        assert 0 < gtol and actual[2] <= gtol and expected[2] <= gtol
        records.append({"log": log.name, "output_sha256": sha(output), "Delta_max_component_error": str(error), "history_max_relative_error": str(history), "residual": str(actual[2]), "gtol": str(gtol), "required_gpu_markers": markers, "acceptance_pass": True})
    return {"candidate": candidate, "live_frozen_hashes_checked": live, "reference_sha256": sha(reference), "input_sha256": sha(source), "records": records, "scope": "Saved J3 g=.202 only; g=.1/.2/.5, idle timing and production integration remain open", "timing_accepted": False}

def check_integration(live=False):
    folder = results / "section9_p3_integrated_consumer"
    archive = json.loads((folder / "archive_metadata.json").read_text())
    assert archive["state"] == "terminal" and archive["exit"] == 0
    for filename, digest in archive["evidence_sha256"].items():
        assert sha(folder / filename) == digest, filename
    assert sha(Path(__file__).with_name("section9_consumer_integrated.patch")) == archive["patch_sha256"]
    manifest = json.loads((folder / "gate_metadata.json").read_text())
    assert manifest["state"] == "terminal" and manifest["exit"] == 0
    assert len(manifest["steps"]) == 10
    if live:
        preparation = json.loads((folder / "preparation_metadata.json").read_text())
        root = Path(preparation["source"])
        for filename, digest in manifest["sha256"].items():
            assert sha(root / filename) == digest, filename
    for step in manifest["steps"]:
        assert step["state"] == "terminal" and step["exit"] == 0
        assert sha(folder / step["log"]) == step["log_sha256"]
        for filename, digest in step.get("artifacts_sha256", {}).items():
            assert sha(folder / step["name"] / filename) == digest, filename
        if step["name"].startswith(("fourier_", "products4_", "lu4_")):
            text = (folder / step["log"]).read_text()
            assert [int(x) for x in re.findall(r"^(\d+) bits:.*pass$", text, re.M)] == [352,374,448]
    expected = reader.read(baseline / "j3_X2Y_g0.202_nc21_cpu.m")
    records = []
    for name in ("default_replay", "combined_replay"):
        combined = name == "combined_replay"
        subfolder = folder / name
        comparison = json.loads((subfolder / "comparison.json").read_text())
        assert len(comparison) == 1 and comparison[0]["acceptance_pass"]
        key = "j3_X2Y_g0.202_nc21"
        cpu = reader.read(subfolder / (key + "_cpu.m"))
        gpu = reader.read(subfolder / (key + "_gpu.m"))
        assert cpu == expected
        assert gpu[1] == cpu[1] and gpu[2] <= Decimal("1e-22")
        with localcontext() as context:
            context.prec = 160
            error = max(abs(x-y) for x,y in zip(cpu[0],gpu[0]))
        assert error <= Decimal("1e-25")
        log = (subfolder / (key + "_gpu.log")).read_text()
        assert "using CPU fallback" not in log and "adjoint check: FAIL" not in log
        assert all(marker in log for marker in ("stage 2 (GPU)","normal matrix (GPU)","first Cholesky (GPU)","adjoint check: PASS"))
        markers = ("Fourier Jacobian (GPU)","exact Fourier:","tangent 4x4 Jacobian (GPU)","tangent products4:","descent inverses4 (GPU)","gluing inverses4 (GPU)","inverse4 LU:")
        assert all(marker in log for marker in markers) if combined else all(marker not in log for marker in markers)
        assert comparison[0]["selected_gpu_switches"] == {"QSC_GPU_NORMAL":False,"QSC_GPU_FOURIER":combined,"QSC_GPU_PRODUCTS4":combined,"QSC_GPU_LU4":combined}
        records.append({"name":name,"CPU_matches_recorded_baseline_exactly":True,"Delta_max_component_error":str(error),"history_exact_serialized_match":True,"residual":str(gpu[2]),"acceptance_pass":True})
    return {"candidate":"integrated_consumer","live_frozen_hashes_checked":live,"records":records,"timing_accepted":False,"scope":"Saved J3 g=.202 only; full requested g sweep, idle timings, damping/base and memory remain open"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    checked = [check(candidate, args.live) for candidate in ("fourier", "small4", "lu4")]
    if (results / "section9_p3_integrated_consumer").exists():
        checked.append(check_integration(args.live))
    print(json.dumps(checked, indent=2))
