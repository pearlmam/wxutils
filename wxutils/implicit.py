# -*- coding: utf-8 -*-

import sympy as sp
import math
import numpy as np

def rotate_implicit_str(func_str: str, angle_deg: float = 45.0) -> str:
    # 1. Define symbolic coordinate variables
    x, y, z = sp.symbols("x y z")

    # 2. Parse input string into a SymPy expression
    expr = sp.sympify(func_str)

    # 3. Compute rotation components
    theta = sp.rad(angle_deg)
    cos_t, sin_t = sp.cos(theta), sp.sin(theta)

    # Inverse transformation coordinates (Z-axis rotation)
    x_inv = x * cos_t + y * sin_t
    y_inv = -x * sin_t + y * cos_t

    # 4. Substitute coordinates into original expression
    rotated_expr = expr.subs({x: x_inv, y: y_inv})

    # 5. Return formatted string (evaluates exact radicals like sqrt(2)/2 to floating points if desired)
    return str(sp.simplify(rotated_expr))

def get_rotated_domain_bounds(
    prob_lo, prob_hi, n_cell, angle_deg, origin=(0.0, 0.0)):
    """Calculates new prob_lo, prob_hi, and n_cell to fit a domain rotated by angle_deg

    around a specified pivot origin [x_0, z_0]. Preserves grid spacing (dx, dz).
    """
    # 1. Unpack first two spatial dimensions
    xmin, zmin = prob_lo[0], prob_lo[1]
    xmax, zmax = prob_hi[0], prob_hi[1]
    nx, nz = n_cell[0], n_cell[1]

    # 2. Compute original cell sizes
    dx = (xmax - xmin) / nx
    dz = (zmax - zmin) / nz

    # 3. Get 4 plane corners and shift relative to pivot origin
    corners = np.array(
        [[xmin, zmin], [xmin, zmax], [xmax, zmin], [xmax, zmax]]
    )
    pivot = np.array(origin)
    centered_corners = corners - pivot

    # 4. Rotate centered corners and shift back to absolute coordinates
    rad = math.radians(angle_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    rotated_corners = (centered_corners @ rot_matrix.T) + pivot

    # 5. Find bounding box extremes
    new_xmin = float(np.min(rotated_corners[:, 0]))
    new_xmax = float(np.max(rotated_corners[:, 0]))
    new_zmin = float(np.min(rotated_corners[:, 1]))
    new_zmax = float(np.max(rotated_corners[:, 1]))

    # 6. Calculate new cell counts with blocking factor alignment
    new_nx = math.ceil((new_xmax - new_xmin) / dx)
    new_nz = math.ceil((new_zmax - new_zmin) / dz)

    # if blocking_factor > 1:
    #     new_nx = math.ceil(new_nx / blocking_factor) * blocking_factor
    #     new_nz = math.ceil(new_nz / blocking_factor) * blocking_factor

    # 7. Adjust upper bounds to maintain exact cell sizes
    adjusted_xmax = new_xmin + new_nx * dx
    adjusted_zmax = new_zmin + new_nz * dz

    new_prob_lo = [new_xmin, new_zmin] + list(prob_lo[2:])
    new_prob_hi = [adjusted_xmax, adjusted_zmax] + list(prob_hi[2:])
    new_n_cell = [new_nx, new_nz] + list(n_cell[2:])

    return new_prob_lo, new_prob_hi, new_n_cell
