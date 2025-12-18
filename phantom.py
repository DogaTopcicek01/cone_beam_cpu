import numpy as np

def _gen_ellipsoid(x_grid, y_grid, z_grid, x0, y0, z0, a, b, c, gray_level, alpha=0, beta=0, gamma=0):
    """
    Returns an image with a 3D ellipsoid in a 3D plane with a center of [x0,y0,z0] and ...

    Args:
        x_grid(float): 3D grid of X coordinate values.
        y_grid(float): 3D grid of Y coordinate values.
        z_grid(float): 3D grid of Z coordinate values.
        x0(float): horizontal center of ellipsoid.
        y0(float): vertical center of ellipsoid.
        z0(float): normal center of ellipsoid.
        a(float): X-axis radius.
        b(float): Y-axis radius.
        c(float): Z-axis radius.
        gray_level(float): Gray level for the ellipse.
        alpha(float): [Default=0.0] counter-clockwise angle of rotation by X-axis in radians.
        beta(float): [Default=0.0] counter-clockwise angle of rotation by Y-axis in radians.
        gamma(float): [Default=0.0] counter-clockwise angle of rotation by Z-axis in radians.

    Return:
        ndarray: 3D array with the same shape as x_grid, y_grid, and z_grid

    """
    # Generate Rotation Matrix.
    rx = np.array([[1, 0, 0], [0, np.cos(-alpha), -np.sin(-alpha)], [0, np.sin(-alpha), np.cos(-alpha)]])
    ry = np.array([[np.cos(-beta), 0, np.sin(-beta)], [0, 1, 0], [-np.sin(-beta), 0, np.cos(-beta)]])
    rz = np.array([[np.cos(-gamma), -np.sin(-gamma), 0], [np.sin(-gamma), np.cos(-gamma), 0], [0, 0, 1]])
    r = np.dot(rx, np.dot(ry, rz))

    cor = np.array([x_grid.flatten() - x0, y_grid.flatten() - y0, z_grid.flatten() - z0])

    image = ((np.dot(r[0], cor)) ** 2 / a ** 2 + (np.dot(r[1], cor)) ** 2 / b ** 2 + (
        np.dot(r[2], cor)) ** 2 / c ** 2 <= 1.0) * gray_level

    return image.reshape(x_grid.shape)


def gen_shepp_logan_3d_raw(num_rows, num_cols, num_slices, scale=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Generate a 3D Shepp Logan phantom with MODIFIED VALUES (good visibility)

    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        num_slices: int, number of slices.
        scale: scaling factor of phantom within the image.
        offset_x: proportion of x-axis offset
        offset_y: proportion of y-axis offset
        offset_z: proportion of z-axis offset

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
    """
    
    # MODIFIED 3D SHEPP-LOGAN VALUES (good visibility!)
    sl3d_paras = [
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 1.0},
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.8},
        {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.2},
        {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.2},
        {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.1},
        {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.1},
        {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.1},
        {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.1},
        {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.1},
        {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.2}
    ]

    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    axis_z = np.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = np.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    shift_x = offset_x * 2.0
    shift_y = offset_y * 2.0
    shift_z = offset_z * 2.0

    for el_paras in sl3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, 
                               x0=el_paras['x0']*scale - shift_x, 
                               y0=el_paras['y0']*scale - shift_y,
                               z0=el_paras['z0']*scale - shift_z,
                               a=el_paras['a']*scale, b=el_paras['b']*scale, c=el_paras['c']*scale,
                               gamma=el_paras['gamma'] / 180.0 * np.pi,
                               gray_level=el_paras['gray_level'])

    return np.transpose(image, (2, 0, 1))


def gen_shepp_logan_3d(num_rows, num_cols, num_slices, block_size=(2,2,2), scale=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Generate a 3D Shepp Logan phantom with MODIFIED VALUES (good visibility)

    Args:
        num_rows: int, number of rows in generated phantom.
        num_cols: int, number of columns in generated phantom.
        num_slices: int, number of slices in generated phantom.
        block_size: (int, int, int): size of block average.
        scale: scaling factor of phantom within the image.
        offset_x: proportion of x-axis offset
        offset_y: proportion of y-axis offset
        offset_z: proportion of z-axis offset

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
    """
    phantom_raw = gen_shepp_logan_3d_raw(num_rows*block_size[1], num_cols*block_size[2], num_slices*block_size[0],
                                         scale=scale, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z)
    phantom = phantom_raw.reshape(phantom_raw.shape[0]//block_size[0], block_size[0],
                                  phantom_raw.shape[1]//block_size[1], block_size[1],
                                  phantom_raw.shape[2]//block_size[2], block_size[2]).sum((1, 3, 5)) / (block_size[0]*block_size[1]*block_size[2])
    return phantom


def nrmse(image, reference_image):
    """
    Compute the normalized root mean square error between image and reference_image.

    Args:
        image: Calculated image
        reference_image: Ground truth image

    Returns:
        Root mean square of (image - reference_image) divided by RMS of reference_image

    """
    rmse = np.sqrt(((image - reference_image) ** 2).mean())
    denominator = np.sqrt(((reference_image) ** 2).mean())

    return rmse/denominator
