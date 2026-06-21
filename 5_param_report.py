"""
Reversible Data Hiding (RDH) via Prediction Error Expansion (PEE)
for 3D Gaussian Splatting Point Clouds — JOURNAL-METRICS REVERSIBILITY
EVALUATION module.

Purpose
-------
Produces a journal-ready CSV report (per recovered file) containing:
  - Payload capacity (parsed from filename, e.g. "50Kb", "10b")
  - Lossless / Distorted status
  - RMSE (geometry/property level)
  - PSNR / SSIM placeholders (render-domain metrics, computed
    separately via an external rasterizer if rendering quality is
    required; this module only certifies *attribute-level* losslessness)
  - MD5 hashes of baseline and recovered files for audit traceability

"""

import os
import re
import csv
import hashlib

import numpy as np
import pandas as pd
from plyfile import PlyData


# =========================================================
# UTILITIES
# =========================================================

class DriveMounter:
    """Mounts Google Drive in a Colab environment if not already mounted."""

    @staticmethod
    def mount(mount_point: str = "/content/drive") -> None:
        try:
            from google.colab import drive
            if not os.path.exists(mount_point):
                drive.mount(mount_point)
                print("[INFO] Google Drive berhasil terhubung.")
            else:
                print("[INFO] Google Drive sudah terhubung sebelumnya.")
        except ImportError:
            print("[INFO] Berjalan di luar Google Colab. Membaca path secara langsung.")


class HashUtils:
    """File hashing helpers for pure binary (MD5) equality checks."""

    @staticmethod
    def compute_md5(file_path: str, chunk_size: int = 65536) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(chunk_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(chunk_size)
        return hasher.hexdigest()


class PayloadSizeParser:
    """Extracts the payload capacity label (e.g. '50Kb', '10b') from a filename."""

    PATTERN = re.compile(r"(\d+Kb|\d+b|\d+KB)")

    @classmethod
    def parse(cls, filename: str) -> str:
        match = cls.PATTERN.search(filename)
        return match.group(1) if match else "Tidak Diketahui"


# =========================================================
# CORE REVERSIBILITY EVALUATION LOGIC (WITH JOURNAL METRICS)
# =========================================================

class JournalMetrics:
    """Small value object holding the journal-ready distortion metrics."""

    def __init__(self, rmse_geometry: str, psnr_render_est: str, ssim_render_est: str):
        self.rmse_geometry = rmse_geometry
        self.psnr_render_est = psnr_render_est
        self.ssim_render_est = ssim_render_est

    @classmethod
    def perfect(cls) -> "JournalMetrics":
        return cls(
            rmse_geometry="0.000000e+00",
            psnr_render_est="Inf (Perfect)",
            ssim_render_est="1.0000",
        )

    @classmethod
    def error(cls) -> "JournalMetrics":
        return cls(rmse_geometry="ERR", psnr_render_est="ERR", ssim_render_est="ERR")

    @classmethod
    def structural_mismatch(cls) -> "JournalMetrics":
        return cls(rmse_geometry="DIFF", psnr_render_est="0.0", ssim_render_est="0.0")

    @classmethod
    def distorted(cls, mean_rmse: float) -> "JournalMetrics":
        return cls(
            rmse_geometry=f"{mean_rmse:.6e}",
            psnr_render_est="Requires Rendering",
            ssim_render_est="Requires Rendering",
        )

    def to_dict(self) -> dict:
        return {
            "RMSE Geometry": self.rmse_geometry,
            "PSNR (Render)": self.psnr_render_est,
            "SSIM (Render)": self.ssim_render_est,
        }


class PlyDeepEvaluator:
    """
    Performs MD5-first, structure-second deep evaluation of a recovered
    .ply against the quantized baseline, returning both a human-readable
    status and journal-ready JournalMetrics.
    """

    def evaluate(self, original_path: str, recovered_path: str):
        """Returns (is_lossless: bool, status: str, metrics: JournalMetrics, details: dict)."""
        hash_orig = HashUtils.compute_md5(original_path)
        hash_rec = HashUtils.compute_md5(recovered_path)

        # CASE 1: Binary-identical (absolute lossless)
        if hash_orig == hash_rec:
            return True, "LOSSLESS (100% Identik secara Biner/MD5)", JournalMetrics.perfect(), {}

        # CASE 2: MD5 differs -> compute internal distortion
        try:
            orig_ply = PlyData.read(original_path)
            rec_ply = PlyData.read(recovered_path)
        except Exception as e:
            return False, f"GAGAL MEMBACA PLY: {e}", JournalMetrics.error(), {}

        if len(orig_ply.elements) != len(rec_ply.elements):
            return (
                False,
                "STRUKTUR BERBEDA (Jumlah elemen tidak sama)",
                JournalMetrics.structural_mismatch(),
                {},
            )

        is_lossless = True
        detected_changes = {}
        all_rmses = []

        # NOTE (bug fix): zip against rec_ply.elements, NOT orig_ply.elements again.
        for orig_el, rec_el in zip(orig_ply.elements, rec_ply.elements):
            orig_props = orig_el.data.dtype.names
            for prop in orig_props:
                orig_arr = orig_el.data[prop]
                rec_arr = rec_el.data[prop]

                if np.array_equal(orig_arr, rec_arr):
                    continue

                is_lossless = False
                if np.issubdtype(orig_arr.dtype, np.number):
                    diff = orig_arr.astype(np.float64) - rec_arr.astype(np.float64)
                    rmse = np.sqrt(np.mean(diff ** 2))
                    all_rmses.append(rmse)
                    detected_changes[f"{orig_el.name}.{prop}"] = f"RMSE: {rmse:.6e}"

        if is_lossless:
            return True, "LOSSLESS (Geometri & Properti 100% Sama Sempurna)", JournalMetrics.perfect(), {}

        mean_rmse = float(np.mean(all_rmses)) if all_rmses else 0.0
        return False, "ADA PERUBAHAN ARTEFAK", JournalMetrics.distorted(mean_rmse), detected_changes


# =========================================================
# RESULT CONTAINER
# =========================================================

class JournalEvaluationResult:
    """Simple data container for one recovered-file journal evaluation row."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# =========================================================
# SINGLE-DATASET BATCH EVALUATION PIPELINE
# =========================================================

class JournalReversibilityPipeline:
    """
    Orchestrates the full journal-metrics reversibility evaluation
    workflow for a single dataset:

      1. Validate baseline .ply and recovered-files folder exist.
      2. Compute the baseline's MD5 once.
      3. Iterate over every recovered .ply file in the folder.
      4. Parse payload capacity from filename.
      5. Run PlyDeepEvaluator (MD5-first, structural-second).
      6. Log results to console.
      7. Export a journal-ready CSV (payload size, status, RMSE, PSNR,
         SSIM placeholders, MD5 baseline/recovered).
    """

    CSV_HEADERS = [
        "No", "Nama File", "Kapasitas Payload", "Status Evaluasi",
        "RMSE Geometry", "PSNR (Render)", "SSIM (Render)",
        "MD5 Baseline", "MD5 Recovered",
    ]

    def __init__(self, path_original: str, folder_recovered: str, path_csv_output: str, dataset_name: str = ""):
        self.path_original = path_original
        self.folder_recovered = folder_recovered
        self.path_csv_output = path_csv_output
        self.dataset_name = dataset_name

        self.evaluator = PlyDeepEvaluator()
        self.results = []

    def _print_banner(self) -> None:
        print("\n" + "=" * 70)
        title = "SISTEM EVALUASI PEMULIHAN COVER 100% (LOSSLESS)"
        if self.dataset_name:
            title += f" — {self.dataset_name}"
        print(title.center(70))
        print("=" * 70)

    def run(self) -> None:
        self._print_banner()

        if not os.path.exists(self.path_original) or not os.path.exists(self.folder_recovered):
            print(f"[ERROR] [{self.dataset_name}] Path tidak ditemukan!")
            print(f"        Baseline : {self.path_original}")
            print(f"        Recovered folder : {self.folder_recovered}")
            return

        file_list = sorted(
            f for f in os.listdir(self.folder_recovered) if f.lower().endswith(".ply")
        )

        if len(file_list) == 0:
            print(f"[WARNING] [{self.dataset_name}] Tidak ditemukan file .ply di folder pemulihan.")
            return

        total_lossless = 0
        md5_baseline = HashUtils.compute_md5(self.path_original)

        for idx, filename in enumerate(file_list, 1):
            is_lossless = self._evaluate_single_file(idx, len(file_list), filename, md5_baseline)
            if is_lossless:
                total_lossless += 1

        print(
            f"\nKesimpulan Akhir [{self.dataset_name}]: "
            f"{total_lossless}/{len(file_list)} file LOSSLESS."
        )

        self._export_csv()

    def _evaluate_single_file(self, idx: int, total: int, filename: str, md5_baseline: str) -> bool:
        path_recovered_file = os.path.join(self.folder_recovered, filename)
        print(f"[{idx}/{total}] Mengevaluasi: {filename}")

        payload_size = PayloadSizeParser.parse(filename)
        md5_recovered = HashUtils.compute_md5(path_recovered_file)

        is_lossless, status, metrics, _details = self.evaluator.evaluate(self.path_original, path_recovered_file)

        if is_lossless:
            print(f"   [LOSSLESS] STATUS: {status}")
            m = metrics.to_dict()
            print(f"      [Metrik Jurnal] RMSE: {m['RMSE Geometry']} | "
                  f"PSNR: {m['PSNR (Render)']} | SSIM: {m['SSIM (Render)']}")
        else:
            print(f"   [DISTORTED] STATUS: {status}")
        print("-" * 60)

        row = JournalEvaluationResult(
            Dataset=self.dataset_name,
            No=idx,
            **{"Nama File": filename},
            **{"Kapasitas Payload": payload_size},
            **{"Status Evaluasi": "LOSSLESS" if is_lossless else "DISTORTED"},
            **metrics.to_dict(),
            **{"MD5 Baseline": md5_baseline},
            **{"MD5 Recovered": md5_recovered},
        )
        self.results.append(row)
        return is_lossless

    def _export_csv(self) -> None:
        os.makedirs(os.path.dirname(self.path_csv_output), exist_ok=True)

        with open(self.path_csv_output, mode="w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            for r in self.results:
                d = r.to_dict()
                writer.writerow({k: d.get(k, "") for k in self.CSV_HEADERS})

        print(f"\n[INFO] CSV Berhasil Diperbarui dengan kolom RMSE, PSNR, SSIM!")
        print(f"[INFO] Path: {self.path_csv_output}")


# =========================================================
# MULTI-DATASET ORCHESTRATOR
# =========================================================

class MultiDatasetJournalEvaluationRunner:
    """
    Runs the JournalReversibilityPipeline sequentially across multiple
    3DGS datasets (e.g., TRAIN, TRUCK, CAR, TOASTER), each with its own
    quantized baseline, recovered-files folder, and CSV output path.
    """

    def __init__(self, dataset_configs: list):
        self.dataset_configs = dataset_configs
        self.all_results = []

    def run_all(self) -> None:
        for cfg in self.dataset_configs:
            name = cfg["name"]
            print("\n" + "#" * 100)
            print(f"# JOURNAL EVALUATION FOR DATASET: {name}".ljust(99) + "#")
            print("#" * 100)

            pipeline = JournalReversibilityPipeline(
                path_original=cfg["path_original"],
                folder_recovered=cfg["folder_recovered"],
                path_csv_output=cfg["path_csv_output"],
                dataset_name=name,
            )
            pipeline.run()
            self.all_results.extend(pipeline.results)

        self._save_combined_report()

    def _save_combined_report(self) -> None:
        if not self.all_results:
            print("[WARN] No results collected across datasets; skipping combined report.")
            return

        combined_dir = "/content/drive/MyDrive/ColabNotebooks/output/combined_results/"
        os.makedirs(combined_dir, exist_ok=True)

        df = pd.DataFrame([r.to_dict() for r in self.all_results])
        combined_csv_path = os.path.join(combined_dir, "all_datasets_journal_evaluation.csv")
        df.to_csv(combined_csv_path, index=False)

        print("\n" + "=" * 100)
        print(f"[INFO] All datasets evaluated. Combined journal CSV saved at: {combined_csv_path}")
        print("=" * 100)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    DriveMounter.mount()

    DATASET_CONFIGS = [
        {
            "name": "TRAIN",
            "path_original": "/content/drive/MyDrive/ColabNotebooks/output/TRAIN/quantized_baseline.ply",
            "folder_recovered": "/content/drive/MyDrive/ColabNotebooks/output/TRAIN/batch_results_recovered/",
            "path_csv_output": "/content/drive/MyDrive/ColabNotebooks/output/TRAIN/analisis_evaluasi_pemulihan.csv",
        },
        {
            "name": "TRUCK",
            "path_original": "/content/drive/MyDrive/ColabNotebooks/output/TRUCK/quantized_baseline.ply",
            "folder_recovered": "/content/drive/MyDrive/ColabNotebooks/output/TRUCK/batch_results_recovered/",
            "path_csv_output": "/content/drive/MyDrive/ColabNotebooks/output/TRUCK/analisis_evaluasi_pemulihan.csv",
        },
        {
            "name": "CAR",
            "path_original": "/content/drive/MyDrive/ColabNotebooks/output/CAR/quantized_baseline.ply",
            "folder_recovered": "/content/drive/MyDrive/ColabNotebooks/output/CAR/batch_results_recovered/",
            "path_csv_output": "/content/drive/MyDrive/ColabNotebooks/output/CAR/analisis_evaluasi_pemulihan.csv",
        },
        {
            "name": "TOASTER",
            "path_original": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/quantized_baseline.ply",
            "folder_recovered": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/batch_results_recovered/",
            "path_csv_output": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/analisis_evaluasi_pemulihan.csv",
        },
    ]

    runner = MultiDatasetJournalEvaluationRunner(dataset_configs=DATASET_CONFIGS)
    runner.run_all()
