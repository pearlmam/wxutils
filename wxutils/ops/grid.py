# -*- coding: utf-8 -*-

from pywarpx import picmi,geometry,amr
constants = picmi.constants

def get_grid_cell_sizes():
    """
    Returns a list of cell sizes [dx, dy, dz] based on the 
    current simulation domain and number of cells.
    """
    lo = geometry.prob_lo
    hi = geometry.prob_hi
    n_cell = amr.n_cell
    
    # Calculate spacing: (Domain Length) / (Number of Cells)
    # This works automatically for 1D, 2D, or 3D
    cell_sizes = [(hi[i] - lo[i]) / n_cell[i] for i in range(len(n_cell))]
    
    return cell_sizes

def node_to_cell_centered(data):
    """
    Averages node-centered array data (N+1 points per axis) 
    to cell-centered data (N points per axis).
    """
    if data.ndim == 1:
        return 0.5 * (data[:-1] + data[1:])
    elif data.ndim == 2:
        return 0.25 * (
            data[:-1, :-1] + data[1:, :-1] + 
            data[:-1, 1:]  + data[1:, 1:]
        )
    elif data.ndim == 3:
        return 0.125 * (
            data[:-1, :-1, :-1] + data[1:, :-1, :-1] +
            data[:-1, 1:, :-1]  + data[1:, 1:, :-1]  +
            data[:-1, :-1, 1:]  + data[1:, :-1, 1:]  +
            data[:-1, 1:, 1:]   + data[1:, 1:, 1:]
        )
    return data
    

def to_cell_centered(data, comp=None, domain_cells=None):
    """
    Universal GPU/CPU converter supporting [x, z] or [x, y, z] array layout.
    Averages only the axes whose length equals target_len + 1 (nodal).
    """
    arr = data.squeeze()
    nodal_axes = []
    # 1. Preferred: Universal shape matching against domain cell counts
    if domain_cells is not None:
        # Match domain_cells dimensionality to squeezed array
        target_cells = [c for c in domain_cells if c > 1] if len(domain_cells) != arr.ndim else domain_cells
        for axis, (curr_len, target_len) in enumerate(zip(arr.shape, target_cells)):
            if curr_len == target_len + 1:
                nodal_axes.append(axis)

    # 2. Fallback: Yee-grid nodal axes assuming [x, z] or [x, y, z] layout
    elif comp is not None:
        comp_key = str(comp).lower().replace("e", "").replace("b", "")
        ndim = arr.ndim

        if ndim == 2:  # Layout: [x, z]
            if comp_key == "x":
                nodal_axes = [1]  # Ex is Nodal in z (axis 1)
            elif comp_key == "z":
                nodal_axes = [0]  # Ez is Nodal in x (axis 0)
            elif comp_key in ("y", "rho", "phi", "node", "scalar"):
                nodal_axes = [0, 1]  # Nodal in both

        elif ndim == 3:  # Layout: [x, y, z]
            if comp_key == "x":
                nodal_axes = [1, 2]  # Nodal in y, z
            elif comp_key == "y":
                nodal_axes = [0, 2]  # Nodal in x, z
            elif comp_key == "z":
                nodal_axes = [0, 1]  # Nodal in x, y
            elif comp_key in ("rho", "phi", "node", "scalar"):
                nodal_axes = [0, 1, 2]

    # 3. GPU/CPU zero-copy slice averaging
    for axis in nodal_axes:
        slc_a = [slice(None)] * arr.ndim
        slc_b = [slice(None)] * arr.ndim
        slc_a[axis] = slice(0, -1)
        slc_b[axis] = slice(1, None)

        arr = 0.5 * (arr[tuple(slc_a)] + arr[tuple(slc_b)])
    return arr

