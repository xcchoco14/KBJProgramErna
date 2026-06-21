"""
Reversible Data Hiding (RDH) via Prediction Error Expansion (PEE)
for 3D Gaussian Splatting Point Clouds — SENDER / EMBEDDING module.

Object-Oriented refactor (English) of the original procedural script.
Note: Please change the path of all folder/file targets.
"""

import os
import re
import time

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree
from skimage.metrics import structural_similarity as ssim


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
    """Pure binary helpers, kept anti-leakage (no text-mode decoding)."""

    @staticmethod
    def bytes_to_bits(byte_data: bytes) -> np.ndarray:
        """Converts raw bytes directly into a bit array."""
        bits = []
        for byte in byte_data:
            bits.extend([int(b) for b in bin(byte)[2:].zfill(8)])
        return np.array(bits)


# =========================================================
# I/O
# =========================================================

class PointCloudIO:
    """Handles loading and exporting 3D Gaussian Splatting (.ply) files."""

    @staticmethod
    def load(path: str):
        plydata = PlyData.read(path)
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
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
        sh_tensor = torch.tensor(sh_dc, dtype=torch.float32)
        return xyz_tensor, sh_tensor, plydata

    @staticmethod
    def export(original_plydata, stego_sh: torch.Tensor, output_path: str) -> None:
        new_vertex = original_plydata["vertex"].data.copy()
        new_vertex["f_dc_0"] = stego_sh[:, 0].detach().cpu().numpy()
        new_vertex["f_dc_1"] = stego_sh[:, 1].detach().cpu().numpy()
        new_vertex["f_dc_2"] = stego_sh[:, 2].detach().cpu().numpy()
        PlyData([PlyElement.describe(new_vertex, "vertex")], text=False).write(output_path)


class PayloadLoader:
    """Loads and naturally sorts binary payload (.txt) files from a folder."""

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
        bit_chars = re.findall(r"[01]", content)
        return np.array([int(b) for b in bit_chars]), path


# =========================================================
# METRICS
# =========================================================

class DistortionMetrics:
    """Computes RMSE, PSNR and SSIM between original and stego color attributes."""

    @staticmethod
    def evaluate(original: torch.Tensor, stego: torch.Tensor):
        orig_np = original.detach().cpu().numpy()
        stego_np = stego.detach().cpu().numpy()

        mse = np.mean((orig_np - stego_np) ** 2)
        rmse = np.sqrt(mse)

        max_val = np.max(orig_np) - np.min(orig_np)
        psnr = 100.0 if mse == 0 else 20 * np.log10(max_val / rmse)

        data_range = orig_np.max() - orig_np.min()
        ssim_val, _ = ssim(orig_np, stego_np, full=True, data_range=data_range, channel_axis=1)

        return rmse, psnr, ssim_val


# =========================================================
# CORE EMBEDDING LOGIC (PEE + CONFLICT MANAGEMENT)
# =========================================================

class PEEEmbedder:
    """
    Reversible Data Hiding embedder using Prediction Error Expansion (PEE).

    Prediction is based on the nearest spatial neighbor (k=2 KD-Tree query).
    A site (target point + channel) is used only if the prediction error is
    below `threshold`, and each point (target or neighbor) is used at most
    once to avoid spatial interference between adjacent embedding sites.
    """

    def __init__(self, scale: int, threshold: int = 20):
        self.scale = scale
        self.threshold = threshold

    def embed(self, xyz: torch.Tensor, sh_attr: torch.Tensor, secret_bits: np.ndarray):
        num_bits_to_hide = len(secret_bits)
        xyz_np = xyz.detach().cpu().numpy()

        tree = KDTree(xyz_np)
        _, neighbor_idx_all = tree.query(xyz_np, k=2)

        sh_stego = torch.round(sh_attr.clone() * self.scale) / self.scale
        bits_embedded = 0
        location_map = []
        used_indices = set()  # protection against spatial neighbor interference

        for idx_target in range(len(xyz)):
            if bits_embedded >= num_bits_to_hide:
                break
            if idx_target in used_indices:
                continue

            idx_neighbor = neighbor_idx_all[idx_target, 1]
            if idx_neighbor in used_indices:
                continue

            for channel in range(3):
                if bits_embedded >= num_bits_to_hide:
                    break

                val_target = int(np.round(sh_stego[idx_target, channel].item() * self.scale))
                val_neighbor = int(np.round(sh_stego[idx_neighbor, channel].item() * self.scale))
                error = val_target - val_neighbor

                if abs(error) < self.threshold:
                    bit = int(secret_bits[bits_embedded])
                    error_prime = 2 * error + bit
                    sh_stego[idx_target, channel] = (val_neighbor + error_prime) / self.scale

                    location_map.append([idx_target, channel])
                    bits_embedded += 1

                    used_indices.add(idx_target)
                    used_indices.add(idx_neighbor)

        return sh_stego, np.array(location_map), bits_embedded


# =========================================================
# RESULT CONTAINER
# =========================================================

class EmbeddingResult:
    """Simple data container for one payload file's embedding result row."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# =========================================================
# BATCH PIPELINE (ORCHESTRATION)
# =========================================================

class BatchEmbeddingPipeline:
    """
    Orchestrates the full sender-side workflow:

      1. Load the cover 3DGS point cloud.
      2. Iterate over every payload (.txt) file.
      3. Embed each payload using PEEEmbedder.
      4. Export the stego .ply and its location map (.npy).
      5. Compute distortion metrics (RMSE / PSNR / SSIM) and scalability
         metrics (embedding rate, embedding speed).
      6. Log results to console and save a master CSV report.
    """

    def __init__(
        self,
        input_ply_path: str,
        payload_folder: str,
        output_dir: str,
        scale: int,
        threshold: int = 50,
        dataset_name: str = "",
    ):
        self.input_ply_path = input_ply_path
        self.payload_folder = payload_folder
        self.output_dir = output_dir
        self.scale = scale
        self.threshold = threshold
        self.dataset_name = dataset_name

        os.makedirs(self.output_dir, exist_ok=True)

        self.payload_loader = PayloadLoader(payload_folder)
        self.embedder = PEEEmbedder(scale=scale, threshold=threshold)

        self.xyz = None
        self.sh_dc = None
        self.ply_original = None
        self.total_points = 0
        self.results = []

    def load_cover(self) -> None:
        print("\n[INFO] Loading original PLY file...")
        self.xyz, self.sh_dc, self.ply_original = PointCloudIO.load(self.input_ply_path)
        self.total_points = len(self.xyz)
        print(f"[INFO] Total points in 3DGS model: {self.total_points} points.")

    def _print_header(self) -> None:
        label = f" DATASET: {self.dataset_name} " if self.dataset_name else ""
        print("\n" + "=" * 130)
        if label:
            print(label.center(130, "="))
        print(
            f"{'Text File':<22} | {'Size (KB)':<9} | {'PSNR (dB)':<10} | {'SSIM':<10} | "
            f"{'Embed Time':<12} | {'Rate (BPP)':<12} | {'Speed (KB/s)':<12} | {'Status'}"
        )
        print("-" * 130)

    def run(self) -> None:
        self.load_cover()
        file_list = self.payload_loader.list_files()

        print(f"[INFO] Found {len(file_list)} text files to evaluate.")
        print(f"[INFO] Sorted file list: {file_list}")

        self._print_header()

        for filename in file_list:
            self._process_single_file(filename)

        print("-" * 130)
        self._save_report()

    def _process_single_file(self, filename: str) -> None:
        base_name = os.path.splitext(filename)[0]
        file_path = os.path.join(self.payload_folder, filename)
        size_kb = os.path.getsize(file_path) / 1024

        secret_bits, _ = self.payload_loader.read_bits(filename)
        total_payload_bits = len(secret_bits)

        if total_payload_bits == 0:
            print(f"[{filename:<20}] | Empty file! Skipping.")
            return

        start_time = time.time()
        stego_sh, location_map, total_embedded = self.embedder.embed(
            self.xyz, self.sh_dc, secret_bits
        )
        elapsed = time.time() - start_time

        embedding_rate_bpp = total_embedded / self.total_points
        embedded_size_kb = (total_embedded / 8) / 1024
        embedding_speed_kb_s = embedded_size_kb / elapsed if elapsed > 0 else 0

        rmse_val, psnr_val, ssim_val = DistortionMetrics.evaluate(self.sh_dc, stego_sh)

        ply_name = f"stego_{base_name}.ply"
        map_name = f"loc_map_{base_name}.npy"
        PointCloudIO.export(self.ply_original, stego_sh, os.path.join(self.output_dir, ply_name))
        np.save(os.path.join(self.output_dir, map_name), location_map)

        status = "SUCCESS" if total_embedded == total_payload_bits else "INSUFFICIENT"

        result = EmbeddingResult(
            Dataset=self.dataset_name,
            Source_File=filename,
            File_Size_KB=round(size_kb, 2),
            Total_Bits_Payload=total_payload_bits,
            Bits_Embedded=total_embedded,
            Embedding_Time_Sec=round(elapsed, 5),
            Embedding_Rate_BPP=round(embedding_rate_bpp, 6),
            Embedding_Speed_KB_s=round(embedding_speed_kb_s, 2),
            RMSE=rmse_val,
            PSNR_dB=psnr_val,
            SSIM=ssim_val,
            Status=status,
        )
        self.results.append(result)

        print(
            f"{filename:<22} | {size_kb:<9.2f} | {psnr_val:<10.4f} | {ssim_val:<10.8f} | "
            f"{elapsed:<10.4f}s | {embedding_rate_bpp:<12.5f} | {embedding_speed_kb_s:<12.2f} | {status}"
        )

    def _save_report(self) -> None:
        df = pd.DataFrame([r.to_dict() for r in self.results])
        csv_path = os.path.join(self.output_dir, "batch_evaluation_dynamic_size.csv")
        df.to_csv(csv_path, index=False)

        print("\n[INFO] Batch evaluation completed successfully!")
        print(f"[INFO] New master CSV with scalability metrics saved at: {csv_path}")


# =========================================================
# MULTI-DATASET ORCHESTRATOR
# =========================================================

class MultiDatasetRunner:
    """
    Runs the BatchEmbeddingPipeline sequentially across multiple 3DGS
    datasets (e.g., TRAIN, TRUCK, CAR, TOASTER), each with its own
    input .ply path and output directory, while sharing a common
    payload folder and scale/threshold configuration.
    """

    def __init__(self, dataset_configs: list, scale: int, threshold: int = 50):
        self.dataset_configs = dataset_configs
        self.scale = scale
        self.threshold = threshold
        self.all_results = []

    def run_all(self) -> None:
        for cfg in self.dataset_configs:
            name = cfg["name"]
            print("\n" + "#" * 130)
            print(f"# STARTING DATASET: {name}".ljust(129) + "#")
            print("#" * 130)

            pipeline = BatchEmbeddingPipeline(
                input_ply_path=cfg["input_ply_path"],
                payload_folder=cfg["payload_folder"],
                output_dir=cfg["output_dir"],
                scale=self.scale,
                threshold=self.threshold,
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
        combined_csv_path = os.path.join(combined_dir, "all_datasets_batch_evaluation.csv")
        df.to_csv(combined_csv_path, index=False)

        print("\n" + "=" * 130)
        print(f"[INFO] All datasets processed. Combined master CSV saved at: {combined_csv_path}")
        print("=" * 130)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    SCALE_PARAMETER = 100000
    THRESHOLD = 50
    PAYLOAD_FOLDER = "/content/drive/MyDrive/ColabNotebooks/Payload/"

    DriveMounter.mount()

    DATASET_CONFIGS = [
        {
            "name": "TRAIN",
            "input_ply_path": "/content/drive/MyDrive/ColabNotebooks/BALL/iteration_30000/point_cloud.ply",
            "payload_folder": PAYLOAD_FOLDER,
            "output_dir": "/content/drive/MyDrive/ColabNotebooks/output/BALL/batch_results_eval/",
        },
        {
            "name": "TRUCK",
            "input_ply_path": "/content/drive/MyDrive/ColabNotebooks/COFFEE/iteration_30000/point_cloud.ply",
            "payload_folder": PAYLOAD_FOLDER,
            "output_dir": "/content/drive/MyDrive/ColabNotebooks/output/COFFEE/batch_results_eval/",
        },
        {
            "name": "CAR",
            "input_ply_path": "/content/drive/MyDrive/ColabNotebooks/CAR/iteration_30000/point_cloud.ply",
            "payload_folder": PAYLOAD_FOLDER,
            "output_dir": "/content/drive/MyDrive/ColabNotebooks/output/CAR/batch_results_eval/",
        },
        {
            "name": "TOASTER",
            "input_ply_path": "/content/drive/MyDrive/ColabNotebooks/TOASTER/iteration_30000/point_cloud.ply",
            "payload_folder": PAYLOAD_FOLDER,
            "output_dir": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/batch_results_eval/",
        },
    ]

    runner = MultiDatasetRunner(
        dataset_configs=DATASET_CONFIGS,
        scale=SCALE_PARAMETER,
        threshold=THRESHOLD,
    )
    runner.run_all()
