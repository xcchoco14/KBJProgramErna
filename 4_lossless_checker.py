"""
Reversible Data Hiding (RDH) via Prediction Error Expansion (PEE)
for 3D Gaussian Splatting Point Clouds — REVERSIBILITY EVALUATION module.

Purpose
-------
Validates the core reversibility claim of the proposed scheme: for every
recovered (.ply) cover produced by the Receiver, this module checks
whether it is bit-for-bit (MD5) identical to the quantized baseline
(the original cover passed through the same float32 round-trip
quantization, but with NO embedding). If MD5 hashes differ, a deeper
element/property-level comparison (including per-property RMSE) is
performed to localize any residual artifact.
"""

import os
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


# =========================================================
# CORE REVERSIBILITY EVALUATION LOGIC
# =========================================================

class PlyReversibilityComparator:
    """
    Compares an original (baseline) .ply against a recovered .ply to
    determine whether cover restoration is perfectly lossless.

    Evaluation proceeds in two stages:
      1. Whole-file MD5 hash comparison (fast path).
      2. If hashes differ, a structural and per-property comparison is
         performed, including RMSE for numeric properties that diverge,
         to localize and quantify any residual distortion.
    """

    def compare(self, original_path: str, recovered_path: str):
        """Returns (is_lossless: bool, status: str, details: dict)."""
        hash_orig = HashUtils.compute_md5(original_path)
        hash_rec = HashUtils.compute_md5(recovered_path)

        if hash_orig == hash_rec:
            return True, "LOSSLESS (100% Identik secara Biner/MD5)", {}

        return self._compare_structure(original_path, recovered_path)

    def _compare_structure(self, original_path: str, recovered_path: str):
        try:
            orig_ply = PlyData.read(original_path)
            rec_ply = PlyData.read(recovered_path)
        except Exception as e:
            return False, f"GAGAL MEMBACA PLY: {e}", {}

        if len(orig_ply.elements) != len(rec_ply.elements):
            return False, "STRUKTUR BERBEDA (Jumlah elemen tidak sama)", {}

        is_lossless = True
        detected_changes = {}

        for orig_el, rec_el in zip(orig_ply.elements, rec_ply.elements):
            element_check = self._compare_element(orig_el, rec_el)
            if element_check is None:
                # Hard structural mismatch -> abort immediately
                return False, self._last_structural_error, {}

            element_lossless, element_changes = element_check
            if not element_lossless:
                is_lossless = False
                detected_changes.update(element_changes)

        if is_lossless:
            return True, "LOSSLESS (Geometri & Properti 100% Sama Sempurna)", {}
        return False, "ADA PERUBAHAN ARTEFAK", detected_changes

    def _compare_element(self, orig_el, rec_el):
        """Returns (is_lossless, changes_dict) for a single PLY element,
        or sets self._last_structural_error and returns None on hard
        structural mismatch (different name / row count / properties)."""

        if orig_el.name != rec_el.name:
            self._last_structural_error = (
                f"STRUKTUR BERBEDA (Nama elemen '{orig_el.name}' vs '{rec_el.name}')"
            )
            return None

        if len(orig_el.data) != len(rec_el.data):
            self._last_structural_error = (
                f"STRUKTUR BERBEDA (Jumlah baris data pada elemen '{orig_el.name}' tidak sama)"
            )
            return None

        orig_props = orig_el.data.dtype.names
        rec_props = rec_el.data.dtype.names

        if orig_props != rec_props:
            self._last_structural_error = (
                f"STRUKTUR BERBEDA (Properti pada elemen '{orig_el.name}' berbeda)"
            )
            return None

        element_lossless = True
        changes = {}

        for prop in orig_props:
            orig_arr = orig_el.data[prop]
            rec_arr = rec_el.data[prop]

            if np.array_equal(orig_arr, rec_arr):
                continue

            element_lossless = False
            key = f"{orig_el.name}.{prop}"
            if np.issubdtype(orig_arr.dtype, np.number):
                diff = orig_arr.astype(np.float64) - rec_arr.astype(np.float64)
                rmse = np.sqrt(np.mean(diff ** 2))
                changes[key] = f"RMSE: {rmse:.6e}"
            else:
                changes[key] = "Ada Perubahan (Non-Numerik)"

        return element_lossless, changes


# =========================================================
# RESULT CONTAINER
# =========================================================

class ReversibilityResult:
    """Simple data container for one recovered-file evaluation result row."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# =========================================================
# SINGLE-DATASET BATCH EVALUATION PIPELINE
# =========================================================

class BatchReversibilityEvaluator:
    """
    Orchestrates the full reversibility-evaluation workflow for a single
    dataset:

      1. Validate that the baseline .ply and the recovered-files folder exist.
      2. Iterate over every recovered .ply file in the folder.
      3. Compare each one against the quantized baseline via
         PlyReversibilityComparator.
      4. Log per-file results to console.
      5. Save a master CSV summary and print a final lossless ratio.
    """

    def __init__(self, path_original: str, folder_recovered: str, dataset_name: str = ""):
        self.path_original = path_original
        self.folder_recovered = folder_recovered
        self.dataset_name = dataset_name

        self.comparator = PlyReversibilityComparator()
        self.results = []

    def _print_banner(self) -> None:
        print("\n" + "=" * 70)
        title = "SISTEM EVALUASI PEMULIHAN COVER 100% (LOSSLESS)"
        if self.dataset_name:
            title += f" — {self.dataset_name}"
        print(title.center(70))
        print("=" * 70)
        print(f"File Original Baseline : {self.path_original}")
        print(f"Folder Hasil Pemulihan : {self.folder_recovered}\n")

    def run(self) -> None:
        self._print_banner()

        if not os.path.exists(self.path_original):
            print("[ERROR] File original (baseline) tidak ditemukan!")
            return

        if not os.path.exists(self.folder_recovered):
            print("[ERROR] Folder hasil pemulihan tidak ditemukan! "
                  "Silakan buat folder baru atau sesuaikan path-nya.")
            return

        file_list = sorted(
            f for f in os.listdir(self.folder_recovered) if f.lower().endswith(".ply")
        )

        if len(file_list) == 0:
            print("[WARNING] Tidak ditemukan file berformat .ply di dalam folder pemulihan.")
            print("Pastikan program Receiver Anda sudah menyimpan hasil pemulihan (.ply) ke folder tersebut.")
            return

        print(f"[INFO] Ditemukan {len(file_list)} file .ply pemulihan untuk dievaluasi.\n")

        total_lossless = 0
        for idx, filename in enumerate(file_list, 1):
            is_lossless = self._evaluate_single_file(idx, len(file_list), filename)
            if is_lossless:
                total_lossless += 1

        print(
            f"\nKesimpulan Akhir [{self.dataset_name}]: "
            f"{total_lossless}/{len(file_list)} file berhasil kembali 100% LOSSLESS."
        )

        self._save_report()

    def _evaluate_single_file(self, idx: int, total: int, filename: str) -> bool:
        path_recovered_file = os.path.join(self.folder_recovered, filename)
        print(f"[{idx}/{total}] Mengevaluasi: {filename}")

        is_lossless, status, details = self.comparator.compare(self.path_original, path_recovered_file)

        if is_lossless:
            print(f"   [LOSSLESS] STATUS: {status}")
        else:
            print(f"   [DISTORTED] STATUS: {status}")
            if details:
                print("      Rincian Sisa Distorsi (Kena Kebocoran/Data Masih Tertinggal):")
                for prop_name, err_val in details.items():
                    print(f"      - {prop_name}: {err_val}")
        print("-" * 60)

        self.results.append(
            ReversibilityResult(
                Dataset=self.dataset_name,
                Recovered_File=filename,
                Is_Lossless=is_lossless,
                Status=status,
                Distortion_Details="; ".join(f"{k}={v}" for k, v in details.items()) if details else "",
            )
        )
        return is_lossless

    def _save_report(self) -> None:
        if not self.results:
            return
        df = pd.DataFrame([r.to_dict() for r in self.results])
        output_dir = os.path.dirname(self.folder_recovered.rstrip("/"))
        csv_path = os.path.join(output_dir, "reversibility_evaluation_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Laporan evaluasi reversibilitas disimpan di: {csv_path}")


# =========================================================
# MULTI-DATASET ORCHESTRATOR
# =========================================================

class MultiDatasetReversibilityRunner:
    """
    Runs the BatchReversibilityEvaluator sequentially across multiple 3DGS
    datasets (e.g., TRAIN, TRUCK, CAR, TOASTER), each with its own
    quantized baseline path and recovered-files folder.
    """

    def __init__(self, dataset_configs: list):
        self.dataset_configs = dataset_configs
        self.all_results = []

    def run_all(self) -> None:
        for cfg in self.dataset_configs:
            name = cfg["name"]
            print("\n" + "#" * 100)
            print(f"# EVALUATING REVERSIBILITY FOR DATASET: {name}".ljust(99) + "#")
            print("#" * 100)

            evaluator = BatchReversibilityEvaluator(
                path_original=cfg["path_original"],
                folder_recovered=cfg["folder_recovered"],
                dataset_name=name,
            )
            evaluator.run()
            self.all_results.extend(evaluator.results)

        self._save_combined_report()
        self._print_overall_summary()

    def _save_combined_report(self) -> None:
        if not self.all_results:
            print("[WARN] No results collected across datasets; skipping combined report.")
            return

        combined_dir = "/content/drive/MyDrive/ColabNotebooks/output/combined_results/"
        os.makedirs(combined_dir, exist_ok=True)

        df = pd.DataFrame([r.to_dict() for r in self.all_results])
        combined_csv_path = os.path.join(combined_dir, "all_datasets_reversibility_report.csv")
        df.to_csv(combined_csv_path, index=False)

        print("\n" + "=" * 100)
        print(f"[INFO] All datasets evaluated. Combined master CSV saved at: {combined_csv_path}")
        print("=" * 100)

    def _print_overall_summary(self) -> None:
        if not self.all_results:
            return

        print("\n" + "=" * 100)
        print("OVERALL SUMMARY — Reversibility Evaluation (All Datasets)")
        print("-" * 100)
        print(f"{'Dataset':<12} | {'Total Files':<12} | {'Lossless':<10} | {'Lossless Ratio'}")
        print("-" * 100)

        df = pd.DataFrame([r.to_dict() for r in self.all_results])
        for dataset_name, group in df.groupby("Dataset", sort=False):
            total = len(group)
            lossless = int(group["Is_Lossless"].sum())
            ratio = f"{lossless}/{total} ({100 * lossless / total:.2f}%)"
            print(f"{dataset_name:<12} | {total:<12} | {lossless:<10} | {ratio}")

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
        },
        {
            "name": "TRUCK",
            "path_original": "/content/drive/MyDrive/ColabNotebooks/output/TRUCK/quantized_baseline.ply",
            "folder_recovered": "/content/drive/MyDrive/ColabNotebooks/output/TRUCK/batch_results_recovered/",
        },
        {
            "name": "CAR",
            "path_original": "/content/drive/MyDrive/ColabNotebooks/output/CAR/quantized_baseline.ply",
            "folder_recovered": "/content/drive/MyDrive/ColabNotebooks/output/CAR/batch_results_recovered/",
        },
        {
            "name": "TOASTER",
            "path_original": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/quantized_baseline.ply",
            "folder_recovered": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/batch_results_recovered/",
        },
    ]

    runner = MultiDatasetReversibilityRunner(dataset_configs=DATASET_CONFIGS)
    runner.run_all()
