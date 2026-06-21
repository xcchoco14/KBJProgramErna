"""
Reversible Data Hiding (RDH) via Prediction Error Expansion (PEE)
for 3D Gaussian Splatting Point Clouds — QUANTIZED BASELINE GENERATOR.

Object-Oriented refactor (English) of the original procedural script.
Mirrors the architecture of the SENDER, EXTRACTION, and RECOVERY modules.

Purpose
-------
Produces a "quantized baseline" .ply for each dataset: the *original*
(un-embedded) cover point cloud, but with its SH DC color channels
passed through the exact same float32 round-trip quantization
(float -> int -> float, scaled by `scale`) used inside the Sender and
Receiver pipelines. This isolates pure quantization distortion from
embedding distortion, providing a fair baseline for RMSE / PSNR / SSIM
comparison against the stego and recovered outputs.
"""

import os

import numpy as np
from plyfile import PlyData


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


# =========================================================
# CORE QUANTIZATION LOGIC
# =========================================================

class Float32Quantizer:
    """
    Simulates the exact float32 quantization round-trip applied to SH DC
    color channels inside the Sender/Receiver pipelines:

        v_int       = round(v_float32 * scale)
        v_recovered = (v_int / scale)  -> cast back to float32

    Applying this to the *original* cover (with no bits embedded)
    isolates the distortion introduced purely by quantization, as
    opposed to distortion introduced by the embedding process itself.
    """

    DC_CHANNELS = ("f_dc_0", "f_dc_1", "f_dc_2")

    def __init__(self, scale: float):
        self.scale = scale

    def quantize_channel(self, values: np.ndarray) -> np.ndarray:
        v_float32 = np.array(values, dtype=np.float32)
        v_int = np.round(v_float32 * self.scale)
        v_recovered = (v_int / self.scale).astype(np.float32)
        return v_recovered

    def quantize_plydata(self, plydata: PlyData) -> PlyData:
        """Quantizes all SH DC channels in-place on the given PlyData object."""
        for dc in self.DC_CHANNELS:
            plydata["vertex"][dc] = self.quantize_channel(plydata["vertex"][dc])
        return plydata


# =========================================================
# I/O
# =========================================================

class QuantizedBaselineIO:
    """Handles loading the raw original .ply and exporting the quantized baseline."""

    @staticmethod
    def load(path_raw_orig: str) -> PlyData:
        return PlyData.read(path_raw_orig)

    @staticmethod
    def export(plydata: PlyData, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plydata.write(output_path)


# =========================================================
# RESULT CONTAINER
# =========================================================

class BaselineResult:
    """Simple data container for one dataset's baseline generation result."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# =========================================================
# SINGLE-DATASET PIPELINE
# =========================================================

class QuantizedBaselinePipeline:
    """
    Orchestrates the quantized-baseline generation workflow for a single dataset:

      1. Load the raw original (un-embedded) cover .ply.
      2. Apply float32 round-trip quantization to all SH DC channels.
      3. Export the quantized baseline .ply.
    """

    def __init__(self, path_raw_orig: str, path_new_baseline: str, scale: float, dataset_name: str = ""):
        self.path_raw_orig = path_raw_orig
        self.path_new_baseline = path_new_baseline
        self.scale = scale
        self.dataset_name = dataset_name

        self.quantizer = Float32Quantizer(scale=scale)

    def run(self) -> BaselineResult:
        print(f"\n[INFO] [{self.dataset_name}] Loading raw original PLY: {self.path_raw_orig}")
        plydata_orig = QuantizedBaselineIO.load(self.path_raw_orig)

        num_points = len(plydata_orig["vertex"].data)
        print(f"[INFO] [{self.dataset_name}] Total points: {num_points}")

        print(f"[INFO] [{self.dataset_name}] Applying float32 round-trip quantization (scale={self.scale})...")
        plydata_quantized = self.quantizer.quantize_plydata(plydata_orig)

        QuantizedBaselineIO.export(plydata_quantized, self.path_new_baseline)

        print(f"[SUCCESS] [{self.dataset_name}] Quantized Baseline Float32 berhasil diperbarui!")
        print(f"[INFO] [{self.dataset_name}] Saved at: {self.path_new_baseline}")

        return BaselineResult(
            Dataset=self.dataset_name,
            Source_PLY=self.path_raw_orig,
            Baseline_PLY=self.path_new_baseline,
            Scale=self.scale,
            Total_Points=num_points,
            Status="SUCCESS",
        )


# =========================================================
# MULTI-DATASET ORCHESTRATOR
# =========================================================

class MultiDatasetBaselineRunner:
    """
    Runs the QuantizedBaselinePipeline sequentially across multiple 3DGS
    datasets (e.g., TRAIN, TRUCK, CAR, TOASTER), each with its own raw
    cover .ply path and quantized-baseline output path, while sharing a
    common scale configuration.
    """

    def __init__(self, dataset_configs: list, scale: float):
        self.dataset_configs = dataset_configs
        self.scale = scale
        self.all_results = []

    def run_all(self) -> None:
        for cfg in self.dataset_configs:
            name = cfg["name"]
            print("\n" + "#" * 100)
            print(f"# GENERATING QUANTIZED BASELINE FOR DATASET: {name}".ljust(99) + "#")
            print("#" * 100)

            pipeline = QuantizedBaselinePipeline(
                path_raw_orig=cfg["path_raw_orig"],
                path_new_baseline=cfg["path_new_baseline"],
                scale=self.scale,
                dataset_name=name,
            )
            result = pipeline.run()
            self.all_results.append(result)

        self._print_summary()

    def _print_summary(self) -> None:
        print("\n" + "=" * 100)
        print("SUMMARY — Quantized Baseline Generation")
        print("-" * 100)
        print(f"{'Dataset':<12} | {'Total Points':<14} | {'Scale':<10} | {'Status':<10} | {'Baseline PLY Path'}")
        print("-" * 100)
        for r in self.all_results:
            d = r.to_dict()
            print(f"{d['Dataset']:<12} | {d['Total_Points']:<14} | {d['Scale']:<10.0f} | {d['Status']:<10} | {d['Baseline_PLY']}")
        print("=" * 100)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    # IMPORTANT: This scale value MUST match the SCALE_PARAMETER used in the
    # Sender / Receiver pipelines, so that quantization distortion is
    # measured under identical conditions.
    SCALE_PARAMETER = 100000.0

    DriveMounter.mount()

    DATASET_CONFIGS = [
        {
            "name": "TRAIN",
            "path_raw_orig": "/content/drive/MyDrive/ColabNotebooks/BALL/iteration_30000/point_cloud.ply",
            "path_new_baseline": "/content/drive/MyDrive/ColabNotebooks/output/BALL/quantized_baseline.ply",
        },
        {
            "name": "TRUCK",
            "path_raw_orig": "/content/drive/MyDrive/ColabNotebooks/COFFEE/iteration_30000/point_cloud.ply",
            "path_new_baseline": "/content/drive/MyDrive/ColabNotebooks/output/COFFEE/quantized_baseline.ply",
        },
        {
            "name": "CAR",
            "path_raw_orig": "/content/drive/MyDrive/ColabNotebooks/CAR/iteration_30000/point_cloud.ply",
            "path_new_baseline": "/content/drive/MyDrive/ColabNotebooks/output/CAR/quantized_baseline.ply",
        },
        {
            "name": "TOASTER",
            "path_raw_orig": "/content/drive/MyDrive/ColabNotebooks/TOASTER/iteration_30000/point_cloud.ply",
            "path_new_baseline": "/content/drive/MyDrive/ColabNotebooks/output/TOASTER/quantized_baseline.ply",
        },
    ]

    runner = MultiDatasetBaselineRunner(
        dataset_configs=DATASET_CONFIGS,
        scale=SCALE_PARAMETER,
    )
    runner.run_all()
