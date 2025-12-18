import torch
import math
import time
import logging

logger = logging.getLogger(__name__)

def compute_sys_matrix(sino_params, img_params, view_angle_list,
                       device='cpu',
                       ISBIJCOMPRESSED=True,
                       ISCIJCOMPRESSED=True,
                       AMATRIX_RHO=1.0):
    """
    Python translation of computeSysMatrix():
      - computeAMatrixParameters
      - allocateSysMatrix
      - computeBMatrix
      - computeCMatrix

    sino_params, img_params are objects with the same fields
    as the C structs.
    view_angle_list is a 1D iterable of beta angles (radians).
    """
    start_time = time.time()

    # This A dict plays the role of struct SysMatrix
    A = {}

    # ===== Part 1: parameters (i_vstride_max, i_wstride_max, N_u, u_0, u_1, scalers) =====
    computeAMatrixParameters(
        sino_params, img_params, A, view_angle_list,
        AMATRIX_RHO=AMATRIX_RHO,
        ISBIJCOMPRESSED=ISBIJCOMPRESSED,
        ISCIJCOMPRESSED=ISCIJCOMPRESSED
    )

    # ===== Part 2: allocate arrays, like allocateSysMatrix() in C =====
    N_x, N_y, N_z = img_params.N_x, img_params.N_y, img_params.N_z
    N_beta = len(view_angle_list)
    i_vstride_max = A['i_vstride_max']
    i_wstride_max = A['i_wstride_max']
    N_u = A['N_u']

    arrays = allocateSysMatrix(N_x, N_y, N_z, N_beta,
                               i_vstride_max, i_wstride_max, N_u,
                               device=device)
    A.update(arrays)

    # ===== Part 3: compute B and C (voxel footprints) =====
    computeBMatrixParameters(
        sino_params, img_params, A, view_angle_list,
        ISBIJCOMPRESSED=ISBIJCOMPRESSED
    )
    computeCMatrix(
        sino_params, img_params, A,
        ISCIJCOMPRESSED=ISCIJCOMPRESSED
    )

    end_time = time.time()
    logging.info(f"compute_sys_matrix finished in {end_time - start_time:.2f} seconds")

    return A


# ======================================================================
#   computeAMatrixParameters  (exactly mirroring the C logic)
# ======================================================================
def computeAMatrixParameters(sino_params, img_params, A, view_angles,
                             AMATRIX_RHO=1.0,
                             ISBIJCOMPRESSED=True,
                             ISCIJCOMPRESSED=True):
    """
    Python translation of computeAMatrixParameters().
    Computes:
      - i_vstride_max, i_wstride_max
      - u_0, u_1, Delta_u, N_u
      - B_ij_max, C_ij_max
      - B_ij_scaler, C_ij_scaler
    and stores them in dict A.
    """

    i_vstride_max = 0
    i_wstride_max = 0
    u_0 = float('inf')
    u_1 = float('-inf')
    B_ij_max = 0.0
    C_ij_max = 0.0

    # ---------------------------
    # Part 1: over x, y, beta (B footprint, i_vstride_max, u_0, u_1, B_ij_max)
    # ---------------------------
    for j_x in range(img_params.N_x):
        x_v = j_x * img_params.Delta_xy + (img_params.x_0 + img_params.Delta_xy / 2.0)
        for j_y in range(img_params.N_y):
            y_v = j_y * img_params.Delta_xy + (img_params.y_0 + img_params.Delta_xy / 2.0)

            for i_beta, beta in enumerate(view_angles):
                cosine = math.cos(beta)
                sine = math.sin(beta)

                # scanner coords
                u_v = cosine * x_v - sine * y_v + sino_params.u_r
                v_v = sine * x_v + cosine * y_v + sino_params.v_r

                # magnification
                M = (sino_params.u_d0 - sino_params.u_s) / (u_v - sino_params.u_s)

                # geometry
                theta = math.atan2(v_v, u_v - sino_params.u_s)
                alpha_xy = beta - theta
                # fold alpha into [-pi/4, pi/4]
                alpha_xy = ((alpha_xy + math.pi/4.0) % (math.pi/2.0)) - math.pi/4.0

                # eq. for W_pv
                W_pv = M * img_params.Delta_xy * math.cos(alpha_xy) / math.cos(theta)

                # i_vstart (Part 1 uses round in C)
                i_vstart = round(
                    (M * v_v - W_pv / 2.0
                     - (sino_params.v_d0 + sino_params.Delta_dv / 2.0))
                    / sino_params.Delta_dv
                )
                i_vstart = max(0, i_vstart)

                # temp_stop
                temp_stop = round(
                    (M * v_v + W_pv / 2.0
                     - (sino_params.v_d0 + sino_params.Delta_dv / 2.0))
                    / sino_params.Delta_dv
                )
                temp_stop = min(temp_stop, sino_params.N_dv - 1)

                # stride in v
                i_vstride = max(temp_stop - i_vstart + 1, 0)
                i_vstride_max = max(i_vstride_max, i_vstride)

                # update u_0, u_1
                u_0 = min(u_0, (u_v - img_params.Delta_xy / 2.0))
                u_1 = max(u_1, (u_v - img_params.Delta_xy / 2.0))

                # For B_ij_max (only if compressed in C)
                if ISBIJCOMPRESSED:
                    delta_v = 0.0
                    # L_v = max{a - max{|b|, c}, 0}
                    #   where a = (W_pv + Delta_dv)/2
                    #         b = (W_pv - Delta_dv)/2
                    L_v = (W_pv - sino_params.Delta_dv) / 2.0
                    L_v = abs(L_v)
                    L_v = max(L_v, delta_v)
                    L_v = (W_pv + sino_params.Delta_dv) / 2.0 - L_v
                    L_v = max(L_v, 0.0)

                    B_ij = img_params.Delta_xy * L_v / (
                        math.cos(alpha_xy) * sino_params.Delta_dv
                    )
                    B_ij_max = max(B_ij_max, B_ij)

    # Delta_u, N_u, updated u_1
    Delta_u = img_params.Delta_xy / AMATRIX_RHO
    N_u = math.ceil((u_1 - u_0) / Delta_u) + 1
    u_1 = u_0 + N_u * Delta_u

    # ---------------------------
    # Part 2: over u, z (C footprint, i_wstride_max, C_ij_max)
    # ---------------------------
    for j_u in range(N_u):
        u_v = j_u * Delta_u + (u_0 + img_params.Delta_xy / 2.0)
        M = (sino_params.u_d0 - sino_params.u_s) / (u_v - sino_params.u_s)
        W_pw = M * img_params.Delta_z

        for j_z in range(img_params.N_z):
            w_v = j_z * img_params.Delta_z + (img_params.z_0 + img_params.Delta_z / 2.0)

            # i_wstart in C:
            #   A->i_wstart = (M*w_v - (w_d0 + Delta_dw/2) - W_pw/2)*(1/Delta_dw) + 0.5;
            i_wstart = int(
                (M * w_v
                 - (sino_params.w_d0 + sino_params.Delta_dw / 2.0)
                 - W_pw / 2.0) * (1.0 / sino_params.Delta_dw)
                + 0.5
            )
            i_wstart = max(i_wstart, 0)

            temp_stop = int(
                (M * w_v
                 - (sino_params.w_d0 + sino_params.Delta_dw / 2.0)
                 + W_pw / 2.0) * (1.0 / sino_params.Delta_dw)
                + 0.5
            )
            temp_stop = min(temp_stop, sino_params.N_dw - 1)

            i_wstride = max(temp_stop - i_wstart + 1, 0)
            i_wstride_max = max(i_wstride_max, i_wstride)

            if ISCIJCOMPRESSED:
                delta_w = 0.0
                # L_w = max{a - max{|b|, c}, 0}
                L_w = (W_pw - sino_params.Delta_dw) / 2.0
                L_w = abs(L_w)
                L_w = max(L_w, delta_w)
                L_w = (W_pw + sino_params.Delta_dw) / 2.0 - L_w
                L_w = max(L_w, 0.0)

                # C_ij
                C_ij = (1.0 / sino_params.Delta_dw) * math.sqrt(
                    1.0 + (w_v * w_v) / ((u_v - sino_params.u_s) ** 2)
                ) * L_w
                C_ij_max = max(C_ij_max, C_ij)

    # Store in A (SysMatrix)
    A['i_vstride_max'] = i_vstride_max
    A['i_wstride_max'] = i_wstride_max
    A['u_0'] = u_0
    A['u_1'] = u_1
    A['Delta_u'] = Delta_u
    A['N_u'] = N_u
    A['B_ij_max'] = B_ij_max
    A['C_ij_max'] = C_ij_max

    if ISBIJCOMPRESSED:
        A['B_ij_scaler'] = B_ij_max / 255.0 if B_ij_max > 0 else 1.0
    else:
        A['B_ij_scaler'] = 1.0

    if ISCIJCOMPRESSED:
        A['C_ij_scaler'] = C_ij_max / 255.0 if C_ij_max > 0 else 1.0
    else:
        A['C_ij_scaler'] = 1.0


# ======================================================================
#   computeBMatrixParameters  (computeBMatrix in C)
# ======================================================================
def computeBMatrixParameters(sino_params, img_params, A, view_angles,
                             ISBIJCOMPRESSED=True):
    """
    Python translation of computeBMatrix().
    Fills:
      - A['i_vstart'], A['i_vstride'], A['j_u'], A['B']
    """

    N_x = img_params.N_x
    N_y = img_params.N_y
    N_beta = len(view_angles)
    i_vstride_max = A['i_vstride_max']

    for j_x in range(N_x):
        x_v = j_x * img_params.Delta_xy + (img_params.x_0 + img_params.Delta_xy / 2.0)
        for j_y in range(N_y):
            y_v = j_y * img_params.Delta_xy + (img_params.y_0 + img_params.Delta_xy / 2.0)

            for i_beta, beta in enumerate(view_angles):
                cosine = math.cos(beta)
                sine = math.sin(beta)

                # scanner coords
                u_v = cosine * x_v - sine * y_v + sino_params.u_r
                v_v = sine * x_v + cosine * y_v + sino_params.v_r

                # magnification
                M = (sino_params.u_d0 - sino_params.u_s) / (u_v - sino_params.u_s)

                # angles
                theta = math.atan2(v_v, u_v - sino_params.u_s)
                alpha_xy = beta - theta
                alpha_xy = ((alpha_xy + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0

                W_pv = M * img_params.Delta_xy * math.cos(alpha_xy) / math.cos(theta)

                # i_vstart (with +0.5 rounding like C)
                i_vstart = int(
                    (M * v_v
                     - W_pv / 2.0
                     - (sino_params.v_d0 + sino_params.Delta_dv / 2.0)
                     ) / sino_params.Delta_dv
                    + 0.5
                )
                i_vstart = max(i_vstart, 0)

                # temp_stop
                temp_stop = int(
                    (M * v_v
                     + W_pv / 2.0
                     - (sino_params.v_d0 + sino_params.Delta_dv / 2.0)
                     ) / sino_params.Delta_dv
                    + 0.5
                )
                temp_stop = min(temp_stop, sino_params.N_dv - 1)

                i_vstride = max(temp_stop - i_vstart + 1, 0)

                A['i_vstart'][j_x, j_y, i_beta] = i_vstart
                A['i_vstride'][j_x, j_y, i_beta] = i_vstride

                # j_u index (integer)
                j_u_val = int(
                    (u_v - (A['u_0'] + img_params.Delta_xy / 2.0)) / A['Delta_u'] + 0.5
                )
                A['j_u'][j_x, j_y, i_beta] = j_u_val

                cosine_alpha = math.cos(alpha_xy)

                # loop over i_v footprint
                for i_v in range(i_vstart, i_vstart + i_vstride):
                    v_d = (sino_params.v_d0 + sino_params.Delta_dv / 2.0) + i_v * sino_params.Delta_dv

                    delta_v = abs(v_d - M * v_v)

                    # L_v = max{ a - max{|b|, c}, 0}
                    L_v = (W_pv - sino_params.Delta_dv) / 2.0
                    L_v = abs(L_v)
                    L_v = max(L_v, delta_v)
                    L_v = (W_pv + sino_params.Delta_dv) / 2.0 - L_v
                    L_v = max(L_v, 0.0)

                    B_ij = img_params.Delta_xy * L_v / (cosine_alpha * sino_params.Delta_dv)

                    idx = i_beta * i_vstride_max + (i_v - i_vstart)

                    if ISBIJCOMPRESSED:
                        # store as compressed, like C
                        val = B_ij / A['B_ij_scaler'] + 0.5
                    else:
                        val = B_ij

                    A['B'][j_x, j_y, idx] = val


# ======================================================================
#   computeCMatrix  (same as C version)
# ======================================================================
def computeCMatrix(sinoParams, imgParams, A, ISCIJCOMPRESSED=True):
    """
    Python translation of computeCMatrix().
    Fills:
      - A['i_wstart'], A['i_wstride'], A['C']
    """

    N_u = A['N_u']
    N_z = imgParams.N_z
    i_wstride_max = A['i_wstride_max']

    for j_u in range(N_u):
        u_v = j_u * A['Delta_u'] + (A['u_0'] + imgParams.Delta_xy / 2.0)
        M = (sinoParams.u_d0 - sinoParams.u_s) / (u_v - sinoParams.u_s)
        W_pw = M * imgParams.Delta_z

        for j_z in range(N_z):
            w_v = j_z * imgParams.Delta_z + (imgParams.z_0 + imgParams.Delta_z / 2.0)

            i_wstart = int(
                (M * w_v
                 - (sinoParams.w_d0 + sinoParams.Delta_dw / 2.0)
                 - W_pw / 2.0) * (1.0 / sinoParams.Delta_dw)
                + 0.5
            )
            i_wstart = max(i_wstart, 0)

            temp_stop = int(
                (M * w_v
                 - (sinoParams.w_d0 + sinoParams.Delta_dw / 2.0)
                 + W_pw / 2.0) * (1.0 / sinoParams.Delta_dw)
                + 0.5
            )
            temp_stop = min(temp_stop, sinoParams.N_dw - 1)

            i_wstride = max(temp_stop - i_wstart + 1, 0)

            A['i_wstart'][j_u, j_z] = i_wstart
            A['i_wstride'][j_u, j_z] = i_wstride

            for i_w in range(i_wstart, i_wstart + i_wstride):
                w_d = (sinoParams.w_d0 + sinoParams.Delta_dw / 2.0) + i_w * sinoParams.Delta_dw
                delta_w = abs(w_d - M * w_v)

                # L_w = max{a - max{|b|, c}, 0}
                L_w = (W_pw - sinoParams.Delta_dw) / 2.0
                L_w = abs(L_w)
                L_w = max(L_w, delta_w)
                L_w = (W_pw + sinoParams.Delta_dw) / 2.0 - L_w
                L_w = max(L_w, 0.0)

                C_ij = (1.0 / sinoParams.Delta_dw) * math.sqrt(
                    1.0 + (w_v * w_v) / ((u_v - sinoParams.u_s) ** 2)
                ) * L_w

                if ISCIJCOMPRESSED:
                    val = C_ij / A['C_ij_scaler'] + 0.5
                else:
                    val = C_ij

                idx = j_z * i_wstride_max + (i_w - i_wstart)
                A['C'][j_u, idx] = val


# ======================================================================
#   allocateSysMatrix  (same dimensions as C multialloc)
# ======================================================================
def allocateSysMatrix(N_x, N_y, N_z, N_beta,
                      i_vstride_max, i_wstride_max, N_u,
                      device='cpu'):
    """
    Python version of allocateSysMatrix().
    Returns a dict of PyTorch tensors with same logical shapes as C arrays.
    """

    A = {}
    # B[j_x][j_y][N_beta * i_vstride_max]
    A['B'] = torch.zeros(
        (N_x, N_y, N_beta * i_vstride_max),
        dtype=torch.float32,
        device=device
    )
    # i_vstart[j_x][j_y][N_beta]
    A['i_vstart'] = torch.zeros(
        (N_x, N_y, N_beta),
        dtype=torch.int32,
        device=device
    )
    # i_vstride[j_x][j_y][N_beta]
    A['i_vstride'] = torch.zeros(
        (N_x, N_y, N_beta),
        dtype=torch.int32,
        device=device
    )
    # j_u[j_x][j_y][N_beta]
    A['j_u'] = torch.zeros(
        (N_x, N_y, N_beta),
        dtype=torch.int32,
        device=device
    )

    # C[j_u][N_z * i_wstride_max]
    A['C'] = torch.zeros(
        (N_u, N_z * i_wstride_max),
        dtype=torch.float32,
        device=device
    )
    # i_wstart[j_u][j_z]
    A['i_wstart'] = torch.zeros(
        (N_u, N_z),
        dtype=torch.int32,
        device=device
    )
    # i_wstride[j_u][j_z]
    A['i_wstride'] = torch.zeros(
        (N_u, N_z),
        dtype=torch.int32,
        device=device
    )

    return A
