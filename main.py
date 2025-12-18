import os
import numpy as np
import torch
import time
from datetime import datetime
import logging
from pathlib import Path
from phantom import gen_shepp_logan_3d
from projections import compute_sys_matrix, forward_project_numba, back_project_numba
from matrix import adam_reconstruction_autograd

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"reconstruction_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================
# HARDCODED PARAMETERS (from your specification)
# ============================================================
num_det_rows = 128                    # Number of detector rows (v)
num_det_channels = 128                 # Number of detector channels (w)
magnification = 2.0                    # (source→detector)/(source→center)
dist_source_detector = 3.0 * num_det_channels  # DSD in ALU
num_views = 64                     # Number of projection views
ITERATIONS = 80                        # Adam iterations
LEARNING_RATE = 0.1                    # Learning rate

# CALCULATED PARAMETERS
Nx = num_det_channels
Ny = num_det_channels  
Nz = num_det_rows
delta_det_channel = 1.0
delta_det_row = 1.0
delta_pixel_image = delta_det_channel / magnification
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)  # ADD THIS

# ============================================================
# HELPER CLASSES
# ============================================================
class ImgParams:
    def __init__(self, Nx, Ny, Nz, delta_pixel_image):
        self.N_x = Nx; self.N_y = Ny; self.N_z = Nz
        self.Delta_xy = delta_pixel_image
        self.Delta_z = delta_pixel_image
        self.x_0 = -self.N_x * self.Delta_xy / 2.0
        self.y_0 = -self.N_y * self.Delta_xy / 2.0
        self.z_0 = -self.N_z * self.Delta_z / 2.0

class SinoParams:
    def __init__(self, dist_source_detector, magnification, num_views, num_det_rows, num_det_channels):
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

    # Random test vectors - FIXED ORDER!
    x = torch.randn(
        (img_params.N_x, img_params.N_y, img_params.N_z),  # CHANGED: (X, Y, Z)
        device=device
    )

    y = torch.randn(
        (sino_params.N_beta,
         sino_params.N_dv,
         sino_params.N_dw),
        device=device
    )

    # Forward and backward
    Ax = forward_project_numba(
        x, A, img_params, sino_params, device=device
    )

    ATy = back_project_numba(
        y, A, img_params, sino_params, device=device
    )

    # Inner products
    lhs = torch.sum(Ax * y)
    rhs = torch.sum(x * ATy)

    rel_err = torch.abs(lhs - rhs) / torch.abs(lhs)

    logger.info(f"  <Ax, y>  = {lhs.item():.6e}")
    logger.info(f"  <x, Aᵀy> = {rhs.item():.6e}")
    logger.info(f"  Relative error = {rel_err.item():.3e}")

    if rel_err < 1e-5:
        logger.info(" PASSED adjointness test")
    else:
        logger.info("FAILED adjointness test")

    return rel_err.item()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--output_dir', type=str, default='frontier_output')
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    
    # Override defaults
    global ITERATIONS, LEARNING_RATE
    ITERATIONS = args.iterations
    LEARNING_RATE = args.lr
    
    logger.info("=" * 60)
    logger.info("3D Cone-Beam CT Reconstruction with Shepp-Logan")
    logger.info("=" * 60)
    
    # Setup device - FIXED for Frontier
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    try:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        logger.info("GPU: AMD ROCm (Frontier) or CPU mode")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nOutput directory: {output_dir.absolute()}")
    
    logger.info(f"\nProblem size:")
    logger.info(f"  Volume: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} voxels")
    logger.info(f"  Sinogram: {num_views} views x {num_det_rows} rows x {num_det_channels} channels")
    
    # ========== 1. GENERATE SHEPP-LOGAN PHANTOM ==========
    logger.info("\n" + "-" * 40)
    logger.info("1. Generating 3D Shepp-Logan phantom...")
    start_time = time.time()
    
    # Generate phantom
    phantom_np = gen_shepp_logan_3d(
        Ny, Nx, Nz,  # Note: order is rows, cols, slices
        block_size=(2, 2, 2), 
        scale=1.0,
    )
    
    # Scale to physical attenuation (same as mbircone demo)
    # New: Transpose to (X, Y, Z) for the matrix
    phantom_np = np.transpose(phantom_np, (2, 1, 0))  # CHANGED: (Z,Y,X) -> (X,Y,Z)

    # Convert to torch tensor - now it's already in (X, Y, Z) order
    phantom = torch.from_numpy(phantom_np).float().to(device)
    
    
    
    phantom_time = time.time() - start_time
    logger.info(f"   ✓ Created Shepp-Logan phantom ({Nz}x{Ny}x{Nx}) in {phantom_time:.2f}s")
    logger.info(f"   Phantom range: [{phantom.min():.4f}, {phantom.max():.4f}]")
    logger.info(f"   Mean intensity: {phantom.mean():.4f}")
    
    # ========== 2. SETUP PARAMETERS ==========
    logger.info("\n2. Setting up parameters...")
    img_params = ImgParams(Nx, Ny, Nz, delta_pixel_image)
    sino_params = SinoParams(dist_source_detector, magnification, num_views, num_det_rows, num_det_channels)
    
    # ========== 3. COMPUTE SYSTEM MATRIX ==========
    logger.info("\n3. Computing system matrix...")
    start_time = time.time()
    
    A = compute_sys_matrix(sino_params, img_params, angles, device=device)
    
    matrix_time = time.time() - start_time
    logger.info(f"   ✓ System matrix computed in {matrix_time:.2f}s")
    logger.info(f"   i_vstride_max = {A['i_vstride_max']}, i_wstride_max = {A['i_wstride_max']}")
    
    # ========== 4. ADJOINTNESS TEST ==========
    logger.info("\n4. Running adjointness test...")
    rel_err = adjointness_test(A, img_params, sino_params, device)
    
    # ========== 5. FORWARD PROJECTION ==========
    logger.info("\n5. Forward projection...")
    start_time = time.time()
    
    sino = forward_project_numba(phantom, A, img_params, sino_params, device=device)
    
    proj_time = time.time() - start_time
    logger.info(f"   ✓ Forward projection completed in {proj_time:.2f}s")
    
  
    # Save phantom and sinogram
    np.save(output_dir / 'phantom.npy', phantom_np)
    np.save(output_dir / 'sinogram.npy', sino_np)
    logger.info(f"   ✓ Saved phantom.npy and sinogram.npy")
    
    # ========== 6. RECONSTRUCTION ==========
    logger.info(f"\n6. ADAM Reconstruction ({ITERATIONS} iterations, lr={LEARNING_RATE})...")
    start_time = time.time()
    
    recon, loss_history = adam_reconstruction_autograd(
        sino=sino,
        A=A,
        img_params=img_params,
        sino_params=sino_params,
        num_iter=ITERATIONS,
        lr=LEARNING_RATE,
        device=device
    )
    
    recon_time = time.time() - start_time
    logger.info(f"   ✓ Reconstruction completed in {recon_time:.2f}s")
    logger.info(f"   Final loss: {loss_history[-1]:.6e}")
    
    # ========== 7. VISUALIZE RECONSTRUCTION ==========
    logger.info("\n7. Visualizing reconstruction...")
    recon_np = recon.cpu().numpy()
    
 
    # ========== 8. SAVE RESULTS ==========
    logger.info("\n8. Saving results...")
    
    # Save reconstruction
    np.save(output_dir / 'reconstruction.npy', recon_np)
    np.save(output_dir / 'loss_history.npy', np.array(loss_history))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RECONSTRUCTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {phantom_time + matrix_time + proj_time + recon_time:.2f}s")
    logger.info(f"\nPerformance:")
    logger.info(f"  Phantom generation: {phantom_time:.2f}s")
    logger.info(f"  System matrix:      {matrix_time:.2f}s")
    logger.info(f"  Forward projection: {proj_time:.2f}s")
    logger.info(f"  Reconstruction:     {recon_time:.2f}s")
    
    logger.info(f"\nFiles saved to {output_dir}/:")
    logger.info(f"  phantom.npy          - {phantom_np.shape}")
    logger.info(f"  sinogram.npy         - {sino_np.shape}")
    logger.info(f"  reconstruction.npy   - {recon_np.shape}")
    logger.info(f"  loss_history.npy     - {len(loss_history)} values")
    
    # Show memory usage
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"\nGPU Memory usage: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    logger.info("\nTo download results:")
    logger.info(f"  scp $USER@frontier.olcf.ornl.gov:{output_dir.absolute()}/*.npy .")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
