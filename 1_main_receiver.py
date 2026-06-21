"""
Reversible Data Hiding (RDH) via Prediction Error Expansion (PEE)
for 3D Gaussian Splatting Point Clouds — RECEIVER / EXTRACTION module.

Object-Oriented refactor (English) of the original procedural script.
Mirrors the architecture of the SENDER / EMBEDDING module.
"""

import os
import re
import time
import hashlib

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData
from scipy.spatial import KDTree


# =========================================================
# UTILITIES
# =========================================================

class DriveMounter:
    """Mounts Google Drive in a Colab environment if not already mounted."""

    @staticmethod
    def mount(mount_point: str = "/content/drive") -> None:
        if not os.path.exists(mount_point):
            from google.colab import drive
            drive.mount(mount_point)


class BinaryUtils:
    """Pure binary helpers, kept anti-distortion (no text-mode decoding)."""

    @staticmethod
    def bits_to_bytes(bits) -> bytes:
        """Converts a binary bit array into a pure bytes object."""
        bytelist = []
        usable_len = len(bits) - (len(bits) % 8)
        for i in range(0, usable_len, 8):
            byte_chunk = bits[i:i + 8]
            byte_str = "".join(map(str, byte_chunk))
            bytelist.append(int(byte_str, 2))
        return bytes(bytelist)

    @staticmethod
    def text_to_bits(text: str) -> np.ndarray:
        """Extracts binary characters ('0'/'1') from raw text content."""
        bit_chars = re.findall(r"[01]", text)
        return np.array([int(b) for b in bit_chars])


# =========================================================
# I/O
# =========================================================

class StegoPointCloudIO:
    """Handles loading stego 3D Gaussian Splatting (.ply) files and location maps."""

    @staticmethod
    def load(ply_path: str, map_path: str):
        plydata = PlyData.read(ply_path)
        xyz = np.stack(
            (plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]),
            axis=-1,
        )
        sh_dc = np.stack(
            (
                plydata["vertex"]["f_dc_0"],
                plydata["vertex"]["f_dc_1"],
                plydata["vertex"]["f_dc_2"],
            ),
            axis=-1,
        )
        loc_map = np.load(map_path)

        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
        sh_tensor = torch.tensor(sh_dc, dtype=torch.float32)
        return xyz_tensor, sh_tensor, loc_map


class PayloadLoader:
    """Loads and naturally sorts original payload (.txt) files from a folder,
    used as ground-truth reference for integrity validation."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    @staticmethod
    def _extract_number(filename: str) -> int:
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else 0

    def list_files(self) -> list:
        files = [f for f in os.listdir(self.folder_path) if f.endswith(".txt")]
        return sorted(files, key=self._extract_number)

    def read_bits(self, filename: str):
        path = os.path.join(self.folder_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return BinaryUtils.text_to_bits(content), path


# =========================================================
# CORE EXTRACTION LOGIC (PEE INVERSE)
# =========================================================

class PEEExtractor:
    """
    Reversible Data Hiding extractor — inverse of PEEEmbedder.

    Reconstructs the same spatial target-neighbor pairing used by the
    sender (k=2 KD-Tree query) for every target index recorded in the
    location map, then recovers each embedded bit via the parity
    (modulo-2) operation on the modified prediction error.
    """

    def __init__(self, scale: int):
        self.scale = scale

    def extract(self, xyz: torch.Tensor, sh_stego: torch.Tensor, loc_map: np.ndarray) -> np.ndarray:
        xyz_np = xyz.detach().cpu().numpy()
        tree = KDTree(xyz_np)

        # 1. Reconstruct spatial neighbor relations using KDTree (batch mode)
        target_indices = loc_map[:, 0]
        _, neighbor_idx_list = tree.query(xyz_np[target_indices], k=2)
        all_neighbors = neighbor_idx_list[:, 1]

        # 2. Move indices into PyTorch tensor space to avoid rounding drift
        device = sh_stego.device
        t_target = torch.tensor(target_indices, dtype=torch.long, device=device)
        t_neighbor = torch.tensor(all_neighbors, dtype=torch.long, device=device)
        t_channel = torch.tensor(loc_map[:, 1], dtype=torch.long, device=device)

        # 3. Fetch float color values directly from the stego tensor (preserve float32 accuracy)
        val_stego_float = sh_stego[t_target, t_channel]
        val_neighbor_float = sh_stego[t_neighbor, t_channel]

        # 4. Quantize via torch.round (must exactly match the Sender's rounding)
        val_stego_quant = torch.round(val_stego_float * self.scale).long()
        val_neighbor_quant = torch.round(val_neighbor_float * self.scale).long()

        # 5. Compute the modified prediction error and extract bit parity
        error_prime = val_stego_quant - val_neighbor_quant
        bits_tensor = torch.abs(error_prime) % 2

        return bits_tensor.cpu().numpy()


# =========================================================
# VALIDATION / INTEGRITY CHECK
# =========================================================

class IntegrityValidator:
    """Compares original vs. extracted binary payloads and computes integrity metrics."""

    @staticmethod
    def compare_binaries(original_bytes: bytes, extracted_bytes: bytes):
        """Performs a pure binary comparison and returns (mismatch_count, accuracy_pct)."""
        if original_bytes == extracted_bytes:
            return 0, 100.0

        mismatch_count = 0
        min_len = min(len(original_bytes), len(extracted_bytes))

        for b1, b2 in zip(original_bytes[:min_len], extracted_bytes[:min_len]):
            if b1 != b2:
                mismatch_count += 1

        mismatch_count += abs(len(original_bytes) - len(extracted_bytes))
        accuracy_pct = ((len(original_bytes) - mismatch_count) / len(original_bytes)) * 100
        return mismatch_count, max(0.0, accuracy_pct)

    @staticmethod
    def classify_status(byte_errors: int, accuracy: float, md5_orig: str, md5_extr: str) -> str:
        if md5_orig == md5_extr and byte_errors == 0:
            return "PERFECT (LOSSLESS)"
        elif byte_errors > 0 and accuracy > 0:
            return "MISMATCH (DISTORTED)"
        else:
            return "FAILED (CORRUPTED)"

    @classmethod
    def validate(cls, original_bytes: bytes, extracted_bytes: bytes) -> dict:
        byte_errors, accuracy = cls.compare_binaries(original_bytes, extracted_bytes)

        md5_orig = hashlib.md5(original_bytes).hexdigest()
        md5_extr = hashlib.md5(extracted_bytes).hexdigest()

        status = cls.classify_status(byte_errors, accuracy, md5_orig, md5_extr)

        return {
            "Original_Bytes": len(original_bytes),
            "Extracted_Bytes": len(extracted_bytes),
            "Byte_Errors": byte_errors,
            "Accuracy_Percentage": round(accuracy, 4),
            "Original_MD5": md5_orig,
            "Extracted_MD5": md5_extr,
            "Status": status,
        }


# =========================================================
# RESULT CONTAINER
# =========================================================

class ExtractionResult:
    """Simple data container for one payload file's extraction/validation result row."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# =========================================================
# BATCH PIPELINE (ORCHESTRATION)
# =========================================================

class BatchExtractionPipeline:
    """
    Orchestrates the full receiver-side workflow:

      1. Iterate over every original payload (.txt) file as ground truth.
      2. Locate the corresponding stego .ply and location map (.npy).
      3. Extract embedded bits using PEEExtractor.
      4. Compare extracted bytes vs. original bytes (MD5 + byte-level diff).
      5. Benchmark extraction time / speed.
      6. Log results to console and save a master CSV integrity report.
    """

    def __init__(
        self,
        payload_dir: str,
        input_dir: str,
        output_summary_dir: str,
        scale: int,
        dataset_name: str = "",
    ):
        self.payload_dir = payload_dir
        self.input_dir = input_dir
        self.output_summary_dir = output_summary_dir
        self.scale = scale
        self.dataset_name = dataset_name

        os.makedirs(self.output_summary_dir, exist_ok=True)

        self.payload_loader = PayloadLoader(payload_dir)
        self.extractor = PEEExtractor(scale=scale)

        self.results = []

    def _print_header(self) -> None:
        label = f" DATASET: {self.dataset_name} " if self.dataset_name else ""
        print("=" * 160)
        if label:
            print(label.center(160, "="))
        print(
            f"{'Nama File Original':<22} | {'Byte Asli':<10} | {'Byte Hasil':<10} | "
            f"{'Error (Byte)':<12} | {'Akurasi (%)':<12} | {'Extract Time':<12} | "
            f"{'Speed (KB/s)':<12} | {'Integrity Status'}"
        )
        print("-" * 160)

    def run(self) -> None:
        file_list = self.payload_loader.list_files()
        self._print_header()

        for filename in file_list:
            self._process_single_file(filename)

        print("=" * 160)
        self._save_report()

    def _process_single_file(self, filename: str) -> None:
        base_identifier = os.path.splitext(filename)[0]

        ply_name = f"stego_{base_identifier}.ply"
        map_name = f"loc_map_{base_identifier}.npy"

        path_original = os.path.join(self.payload_dir, filename)
        path_ply = os.path.join(self.input_dir, ply_name)
        path_map = os.path.join(self.input_dir, map_name)

        if not os.path.exists(path_ply) or not os.path.exists(path_map):
            print(f"{filename:<22} | [WARNING] STEGO FILE / LOC MAP NOT FOUND IN input_dir]")
            return

        xyz, sh_stego, loc_map = StegoPointCloudIO.load(path_ply, path_map)

        # Start time benchmarking
        start_time = time.time()
        bits = self.extractor.extract(xyz, sh_stego, loc_map)
        extracted_bytes = BinaryUtils.bits_to_bytes(bits)
        end_time = time.time()

        extraction_time_sec = end_time - start_time
        extracted_size_kb = len(extracted_bytes) / 1024
        extraction_speed_kb_s = extracted_size_kb / extraction_time_sec if extraction_time_sec > 0 else 0

        original_bits, _ = self.payload_loader.read_bits(filename)

        # Synchronize bit length if payload exceeds location map capacity
        if len(original_bits) > len(bits):
            original_bits = original_bits[: len(bits)]

        original_bytes = BinaryUtils.bits_to_bytes(original_bits)

        validation = IntegrityValidator.validate(original_bytes, extracted_bytes)

        print(
            f"{filename:<22} | {validation['Original_Bytes']:<10} | {validation['Extracted_Bytes']:<10} | "
            f"{validation['Byte_Errors']:<12} | {validation['Accuracy_Percentage']:<12.4f} | "
            f"{extraction_time_sec:<10.4f}s | {extraction_speed_kb_s:<12.2f} | {validation['Status']}"
        )

        result = ExtractionResult(
            Dataset=self.dataset_name,
            File_Name=filename,
            Original_Bytes=validation["Original_Bytes"],
            Extracted_Bytes=validation["Extracted_Bytes"],
            Byte_Errors=validation["Byte_Errors"],
            Accuracy_Percentage=validation["Accuracy_Percentage"],
            Extraction_Time_Sec=round(extraction_time_sec, 5),
            Extraction_Speed_KB_s=round(extraction_speed_kb_s, 2),
            Original_MD5=validation["Original_MD5"],
            Extracted_MD5=validation["Extracted_MD5"],
            Status=validation["Status"],
        )
        self.results.append(result)

    def _save_report(self) -> None:
        df = pd.DataFrame([r.to_dict() for r in self.results])
        csv_path = os.path.join(self.output_summary_dir, "master_integrity_report.csv")
        df.to_csv(csv_path, index=False)

        print(f"\n[INFO] Proses Selesai Semuanya untuk dataset '{self.dataset_name}'!")
        print(f"[INFO] Laporan komparasi final biner + metrik waktu disimpan di: {csv_path}")


# =========================================================
# MULTI-DATASET ORCHESTRATOR
# =========================================================

class MultiDatasetExtractionRunner:
    """
    Runs the BatchExtractionPipeline sequentially across multiple 3DGS
    datasets (e.g., TRAIN, TRUCK, CAR, TOASTER), each with its own
    stego input directory and output summary directory, while sharing
    a common payload (ground-truth) folder and scale configuration.
    """

    def __init__(self, dataset_configs: list, scale: int):
        self.dataset_configs = dataset_configs
        self.scale = scale
        self.all_results = []

    def run_all(self) -> None:
        for cfg in self.dataset_configs:
            name = cfg["name"]
            print("\n" + "#" * 160)
            print(f"# STARTING EXTRACTION FOR DATASET: {name}".ljust(159) + "#")
            print("#" * 160)

            pipeline = BatchExtractionPipeline(
                payload_dir=cfg["payload_dir"],
                input_dir=cfg["input_dir"],
                output_summary_dir=cfg["output_summary_dir"],
                scale=self.scale,
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
        combined_csv_path = os.path.join(combined_dir, "all_datasets_integrity_report.csv")
        df.to_csv(combined_csv_path, index=False)

        print("\n" + "=" * 160)
        print(f"[INFO] All datasets extracted & validated. Combined master CSV saved at: {combined_csv_path}")
        print("=" * 160)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    # IMPORTANT: This scale value MUST match the SCALE_PARAMETER used in the Sender.
    SCALE_PARAMETER = 100000
    PAYLOAD_FOLDER = "/content/drive/MyDrive/ColabNotebooks/Payload/"

    DriveMounter.mount()

    DATASET_CONFIGS = [
        {
            "name": "TRAIN",
            "payload_dir": PAYLOAD_FOLDER,
            "input_dir": "/content/drive/MyDrive/ColabNotebooks/output/TRAIN/batch_results_eval/",
            "output_summary_dir": "/content/drive/MyDrive/ColabNotebooks/output/TRAIN",
        },
        {
            "name": "TRUCK",
            "payload_dir": PAYLOAD_FOLDER,
            "input_dir": "/content/drive/MyDrive/ColabNotebooks/output/TRUCK/batch_results_eval",
            "output_summary_dir": "/content/drive/MyDrive/ColabNotebooks/output/TRUCK/",
        },
        {
            "name": "CAR",
            "payload_dir": PAYLOAD_FOLDER,
            "input_dir": "/content/drive/MyDrive/ColabNotebooks/output/CAR/batch_results_eval",
            "output_summary_dir": "/content/drive/MyDrive/ColabNotebooks/output/CAR/",
        },
        {
            "name": "TOASTER",
            "payload_dir": PAYLOAD_FOLDER,
            "input_dir": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/batch_results_eval",
            "output_summary_dir": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/",
        },
    ]

    runner = MultiDatasetExtractionRunner(
        dataset_configs=DATASET_CONFIGS,
        scale=SCALE_PARAMETER,
    )
    runner.run_all()
