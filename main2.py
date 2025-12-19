import os
import numpy as np
import torch
import time
from datetime import datetime
import logging
from pathlib import Path

from phantom import gen_shepp_logan_3d
from functions import forward_project_numba, back_project_numba
from matrix import compute_sys_matrix
from gd_adam import adam_reconstruction_autograd



# ============================================================
# HELPER CLASSES
# ============================================================
class ImgParams:
    def __init__(self, Nx, Ny, Nz, delta_pixel_image):
        self.N_x = Nx
        self.N_y = Ny
        self.N_z = Nz
        self.Delta_xy = delta_pixel_image
        self.Delta_z = delta_pixel_image
        self.x_0 = -self.N_x * self.Delta_xy / 2.0
        self.y_0 = -self.N_y * self.Delta_xy / 2.0
        self.z_0 = -self.N_z * self.Delta_z / 2.0


class SinoParams:
    def __init__(self, dist_source_detector, magnification,
                 num_views, num_det_rows, num_det_channels):
        self.N_dv = num_det_rows
        self.N_dw = num_det_channels
        self.N_beta = num_views
        self.Delta_dv = 1.0
        self.Delta_dw = 1.0
        self.u_s = -dist_source_detector / magnification
        self.u_r = 0.0
        self.v_r = 0.0
        self.u_d0 = dist_source_detector - dist_source_detector / magnification
        self.v_d0 = -self.N_dv * self.Delta_dv / 2.0
        self.w_d0 = -self.N_dw * self.Delta_dw / 2.0


# ============================================================
# ADJOINTNESS TEST
# ============================================================
def adjointness_test(A, img_params, sino_params, device):
    logger.info("-" * 40)
    logger.info("Adjointness test ⟨Ax, y⟩ vs ⟨x, Aᵀy⟩")

    x = torch.randn(
        (img_params.N_x, img_params.N_y, img_params.N_z),
        device=device
    )

    y = torch.randn(
        (sino_params.N_beta, sino_params.N_dv, sino_params.N_dw),
        device=device
    )

    Ax = forward_project_numba(x, A, img_params, sino_params, device=device)
    ATy = back_project_numba(y, A, img_params, sino_params, device=device)

    lhs = torch.sum(Ax * y)
    rhs = torch.sum(x * ATy)
    rel_err = torch.abs(lhs - rhs) / torch.abs(lhs)

    logger.info(f"<Ax, y>  = {lhs.item():.6e}")
    logger.info(f"<x, Aᵀy> = {rhs.item():.6e}")
    logger.info(f"Relative error = {rel_err.item():.3e}")

    return rel_err.item()


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="3D Cone-Beam CT Reconstruction with Shepp-Logan"
    )

    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--output_dir", type=str, default="frontier_output")

    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--det_rows", type=int, default=128)
    parser.add_argument("--det_channels", type=int, default=128)
    parser.add_argument("--magnification", type=float, default=2.0)
    parser.add_argument("--views", type=int, default=64)
    parser.add_argument("--dsd_factor", type=float, default=3.0)

    parser.add_argument("--block_size", type=int, nargs=3,
                        default=[2, 2, 2])
    parser.add_argument("--phantom_scale", type=float, default=1.0)
    parser.add_argument("--skip_adjoint_test", action="store_true")

    args = parser.parse_args()

    # ============================================================
    # FORCE OUTPUT NEXT TO main.py
    # ============================================================
    SCRIPT_DIR = Path(__file__).resolve().parent
    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"reconstruction_{timestamp}.log"

    # ============================================================
    # LOGGING
    # ============================================================

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force = True,
    )

    logger = logging.getLogger(__name__)

    logger.info(f"PWD: {os.getcwd()}")
    logger.info(f"SCRIPT_DIR: {SCRIPT_DIR}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")

    # ============================================================
    # PARAMETERS
    # ============================================================
    num_det_rows = args.det_rows
    num_det_channels = args.det_channels
    magnification = args.magnification
    num_views = args.views

    ITERATIONS = args.iterations
    LEARNING_RATE = args.lr

    dist_source_detector = args.dsd_factor * num_det_channels

    Nx = num_det_channels
    Ny = num_det_channels
    Nz = num_det_rows

    delta_pixel_image = 1.0 / magnification
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # ============================================================
    # PHANTOM
    # ============================================================
    phantom_np = gen_shepp_logan_3d(
        Ny, Nx, Nz,
        block_size=tuple(args.block_size),
        scale=args.phantom_scale,
    )
    phantom_np = np.transpose(phantom_np, (2, 1, 0))
    phantom = torch.from_numpy(phantom_np).float().to(device)

    # ============================================================
    # GEOMETRY
    # ============================================================
    img_params = ImgParams(Nx, Ny, Nz, delta_pixel_image)
    sino_params = SinoParams(
        dist_source_detector, magnification,
        num_views, num_det_rows, num_det_channels
    )

    # ============================================================
    # SYSTEM MATRIX
    # ============================================================
    A = compute_sys_matrix(sino_params, img_params, angles, device=device)

    # ============================================================
    # ADJOINT TEST
    # ============================================================
    if not args.skip_adjoint_test:
        adjointness_test(A, img_params, sino_params, device)

    # ============================================================
    # FORWARD PROJECTION
    # ============================================================
    sino = forward_project_numba(
        phantom, A, img_params, sino_params, device=device
    )

    np.save(output_dir / "phantom.npy", phantom_np)
    np.save(output_dir / "sinogram.npy", sino.cpu().numpy())

    # ============================================================
    # RECONSTRUCTION
    # ============================================================
    recon, loss_history = adam_reconstruction_autograd(
        sino=sino,
        A=A,
        img_params=img_params,
        sino_params=sino_params,
        num_iter=ITERATIONS,
        lr=LEARNING_RATE,
        device=device,
    )

    np.save(output_dir / "reconstruction.npy", recon.cpu().numpy())
    np.save(output_dir / "loss_history.npy", np.array(loss_history))

    logger.info("DONE")
    logger.info(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
