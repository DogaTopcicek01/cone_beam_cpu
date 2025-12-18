import numpy as np
import torch
from numba import jit, prange
import time
import os
from pathlib import Path

# ============================================================
# NUMBA CORE FUNCTIONS
# ============================================================

@jit(nopython=True, parallel=True, fastmath=True)
def forward_project_numba_core(x, B, C, j_u, i_vstart, i_vstride, i_wstart, i_wstride,
                              i_vstride_max, i_wstride_max, B_ij_scaler, C_ij_scaler,
                              N_beta, N_dv, N_dw):
    Nx, Ny, Nz = x.shape
    Ax = np.zeros((N_beta, N_dv, N_dw), dtype=np.float32)

    for i_beta in prange(N_beta):
        base_B_idx = i_beta * i_vstride_max

        for j_x in range(Nx):
            for j_y in range(Ny):
                ju = j_u[j_x, j_y, i_beta]
                if ju < 0:
                    continue

                v0 = i_vstart[j_x, j_y, i_beta]
                vl = i_vstride[j_x, j_y, i_beta]
                if vl <= 0:
                    continue

                for iv_offset in range(vl):
                    i_v = v0 + iv_offset
                    if i_v >= N_dv:
                        break

                    B_idx = base_B_idx + iv_offset
                    B_ij = B_ij_scaler * B[j_x, j_y, B_idx]

                    if B_ij == 0:
                        continue

                    for j_z in range(Nz):
                        x_val = x[j_x, j_y, j_z]
                        if x_val == 0:
                            continue

                        w0 = i_wstart[ju, j_z]
                        wl = i_wstride[ju, j_z]
                        if wl <= 0:
                            continue

                        for iw_offset in range(wl):
                            i_w = w0 + iw_offset
                            if i_w >= N_dw:
                                break

                            C_idx = j_z * i_wstride_max + iw_offset
                            C_ij = C_ij_scaler * C[ju, C_idx]

                            Ax[i_beta, i_v, i_w] += B_ij * x_val * C_ij
    return Ax

@jit(nopython=True, parallel=True, fastmath=True)
def back_project_numba_core(Ax, B, C, j_u, i_vstart, i_vstride, i_wstart, i_wstride,
                           i_vstride_max, i_wstride_max, B_ij_scaler, C_ij_scaler,
                           Nx, Ny, Nz, N_beta):
    x = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    for i_beta in prange(N_beta):
        base_B_idx = i_beta * i_vstride_max

        for j_x in range(Nx):
            for j_y in range(Ny):
                ju = j_u[j_x, j_y, i_beta]
                if ju < 0:
                    continue

                v0 = i_vstart[j_x, j_y, i_beta]
                vl = i_vstride[j_x, j_y, i_beta]
                if vl <= 0:
                    continue

                for iv_offset in range(vl):
                    i_v = v0 + iv_offset
                    if i_v >= Ax.shape[1]:
                        break

                    B_idx = base_B_idx + iv_offset
                    B_ij = B_ij_scaler * B[j_x, j_y, B_idx]

                    if B_ij == 0:
                        continue

                    for j_z in range(Nz):
                        w0 = i_wstart[ju, j_z]
                        wl = i_wstride[ju, j_z]
                        if wl <= 0:
                            continue

                        sum_val = 0.0
                        for iw_offset in range(wl):
                            i_w = w0 + iw_offset
                            if i_w >= Ax.shape[2]:
                                break

                            C_idx = j_z * i_wstride_max + iw_offset
                            C_ij = C_ij_scaler * C[ju, C_idx]

                            A_ij = B_ij * C_ij
                            sum_val += A_ij * Ax[i_beta, i_v, i_w]

                        x[j_x, j_y, j_z] += sum_val
    return x

def forward_project_numba(x, A, img_params, sino_params, device=None):
    # Determine target device
    if device is None:
        device = x.device
    elif isinstance(device, str):
        device = torch.device(device)

    # Move all tensors to CPU for Numba processing
    if x.is_cuda:
        x_cpu = x.cpu()
    else:
        x_cpu = x

    # Ensure A tensors are on CPU
    B_cpu = A["B"].cpu() if torch.is_tensor(A["B"]) and A["B"].is_cuda else A["B"]
    C_cpu = A["C"].cpu() if torch.is_tensor(A["C"]) and A["C"].is_cuda else A["C"]
    j_u_cpu = A["j_u"].cpu() if torch.is_tensor(A["j_u"]) and A["j_u"].is_cuda else A["j_u"]
    i_vstart_cpu = A["i_vstart"].cpu() if torch.is_tensor(A["i_vstart"]) and A["i_vstart"].is_cuda else A["i_vstart"]
    i_vstride_cpu = A["i_vstride"].cpu() if torch.is_tensor(A["i_vstride"]) and A["i_vstride"].is_cuda else A["i_vstride"]
    i_wstart_cpu = A["i_wstart"].cpu() if torch.is_tensor(A["i_wstart"]) and A["i_wstart"].is_cuda else A["i_wstart"]
    i_wstride_cpu = A["i_wstride"].cpu() if torch.is_tensor(A["i_wstride"]) and A["i_wstride"].is_cuda else A["i_wstride"]

    if x_cpu.ndim == 3:
        x_np = x_cpu.numpy()
    else:
        raise ValueError(f"Expected 3D tensor, got shape {x_cpu.shape}")

    # Convert to numpy, but only if they're tensors
    B_np = B_cpu.numpy() if torch.is_tensor(B_cpu) else B_cpu
    C_np = C_cpu.numpy() if torch.is_tensor(C_cpu) else C_cpu
    j_u_np = j_u_cpu.numpy() if torch.is_tensor(j_u_cpu) else j_u_cpu
    i_vstart_np = i_vstart_cpu.numpy() if torch.is_tensor(i_vstart_cpu) else i_vstart_cpu
    i_vstride_np = i_vstride_cpu.numpy() if torch.is_tensor(i_vstride_cpu) else i_vstride_cpu
    i_wstart_np = i_wstart_cpu.numpy() if torch.is_tensor(i_wstart_cpu) else i_wstart_cpu
    i_wstride_np = i_wstride_cpu.numpy() if torch.is_tensor(i_wstride_cpu) else i_wstride_cpu

    i_vstride_max = int(A["i_vstride_max"])
    i_wstride_max = int(A["i_wstride_max"])
    B_ij_scaler = float(A["B_ij_scaler"])
    C_ij_scaler = float(A["C_ij_scaler"])

    N_beta = sino_params.N_beta
    N_dv = sino_params.N_dv
    N_dw = sino_params.N_dw

    Ax_np = forward_project_numba_core(
        x_np, B_np, C_np, j_u_np, i_vstart_np, i_vstride_np,
        i_wstart_np, i_wstride_np, i_vstride_max, i_wstride_max,
        B_ij_scaler, C_ij_scaler, N_beta, N_dv, N_dw
    )

    Ax_torch = torch.from_numpy(Ax_np)

    # Move to target device
    if isinstance(device, torch.device) and device.type == 'cuda':
        return Ax_torch.to(device)
    return Ax_torch

def back_project_numba(Ax, A, img_params, sino_params, device=None):
    # Determine target device
    if device is None:
        device = Ax.device
    elif isinstance(device, str):
        device = torch.device(device)

    # Move all tensors to CPU for Numba processing
    if Ax.is_cuda:
        Ax_cpu = Ax.cpu()
    else:
        Ax_cpu = Ax

    # Ensure A tensors are on CPU
    B_cpu = A["B"].cpu() if torch.is_tensor(A["B"]) and A["B"].is_cuda else A["B"]
    C_cpu = A["C"].cpu() if torch.is_tensor(A["C"]) and A["C"].is_cuda else A["C"]
    j_u_cpu = A["j_u"].cpu() if torch.is_tensor(A["j_u"]) and A["j_u"].is_cuda else A["j_u"]
    i_vstart_cpu = A["i_vstart"].cpu() if torch.is_tensor(A["i_vstart"]) and A["i_vstart"].is_cuda else A["i_vstart"]
    i_vstride_cpu = A["i_vstride"].cpu() if torch.is_tensor(A["i_vstride"]) and A["i_vstride"].is_cuda else A["i_vstride"]
    i_wstart_cpu = A["i_wstart"].cpu() if torch.is_tensor(A["i_wstart"]) and A["i_wstart"].is_cuda else A["i_wstart"]
    i_wstride_cpu = A["i_wstride"].cpu() if torch.is_tensor(A["i_wstride"]) and A["i_wstride"].is_cuda else A["i_wstride"]

    Ax_np = Ax_cpu.numpy()

    # Convert to numpy, but only if they're tensors
    B_np = B_cpu.numpy() if torch.is_tensor(B_cpu) else B_cpu
    C_np = C_cpu.numpy() if torch.is_tensor(C_cpu) else C_cpu
    j_u_np = j_u_cpu.numpy() if torch.is_tensor(j_u_cpu) else j_u_cpu
    i_vstart_np = i_vstart_cpu.numpy() if torch.is_tensor(i_vstart_cpu) else i_vstart_cpu
    i_vstride_np = i_vstride_cpu.numpy() if torch.is_tensor(i_vstride_cpu) else i_vstride_cpu
    i_wstart_np = i_wstart_cpu.numpy() if torch.is_tensor(i_wstart_cpu) else i_wstart_cpu
    i_wstride_np = i_wstride_cpu.numpy() if torch.is_tensor(i_wstride_cpu) else i_wstride_cpu

    i_vstride_max = int(A["i_vstride_max"])
    i_wstride_max = int(A["i_wstride_max"])
    B_ij_scaler = float(A["B_ij_scaler"])
    C_ij_scaler = float(A["C_ij_scaler"])

    Nx = img_params.N_x
    Ny = img_params.N_y
    Nz = img_params.N_z
    N_beta = sino_params.N_beta

    x_np = back_project_numba_core(
        Ax_np, B_np, C_np, j_u_np, i_vstart_np, i_vstride_np,
        i_wstart_np, i_wstride_np, i_vstride_max, i_wstride_max,
        B_ij_scaler, C_ij_scaler, Nx, Ny, Nz, N_beta
    )

    x_torch = torch.from_numpy(x_np).contiguous()

    # Move to target device
    if isinstance(device, torch.device) and device.type == 'cuda':
        return x_torch.to(device)
    return x_torch

# ============================================================
# COMPATIBILITY WRAPPERS
# ============================================================

def forward_project_fast(x, A, img_params, sino_params, device=None):
    """Wrapper for Numba forward projection"""
    return forward_project_numba(x, A, img_params, sino_params, device=device)

def back_project_fast(Ax, A, img_params, sino_params, device=None):
    """Wrapper for Numba back projection"""
    return back_project_numba(Ax, A, img_params, sino_params, device=device)

# ============================================================
# AUTOGRAD FUNCTIONS (for adam_reconstruction_autograd)
# ============================================================

class ConeBeamProjectorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, img_params, sino_params, device=None):
        # Store device information
        ctx.device = device if device is not None else x.device

        # Store A components properly (only convert tensors to CPU, leave scalars as is)
        ctx.A = {}
        for k, v in A.items():
            if torch.is_tensor(v):
                ctx.A[k] = v.to('cpu')
            else:
                ctx.A[k] = v  # Keep scalars as is

        ctx.img_params = img_params
        ctx.sino_params = sino_params

        # Perform projection on CPU (Numba requires CPU)
        x_cpu = x.detach().cpu() if x.is_cuda else x.detach()
        Ax = forward_project_numba(x_cpu, ctx.A, img_params, sino_params, device='cpu')

        # Return result on the same device as input
        return Ax.to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Move gradient to CPU for backprojection
        grad_cpu = grad_output.detach().cpu() if grad_output.is_cuda else grad_output.detach()

        # Perform backprojection on CPU
        grad_x = back_project_numba(grad_cpu, ctx.A, ctx.img_params, ctx.sino_params, device='cpu')

        # Return gradient on the same device as input
        return grad_x.to(ctx.device), None, None, None, None

def cone_beam_projector(x, A, img_params, sino_params, device=None):
    # Convert string device to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = x.device

    return ConeBeamProjectorFn.apply(x, A, img_params, sino_params, device)
