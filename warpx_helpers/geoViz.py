# -*- coding: utf-8 -*-

import numpy as np
import trimesh
from skimage import measure
# needs module shapely, mapbox-earcut

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def heaviside(x1,x2):
    return np.heaviside(x1,x2)

def max(x1,x2=None):
    if x2 is None:
        return np.max(x1)
    else:
        return np.maximum(x1,x2)

def plot_grid(x,z):
    plt.plot(x, z, color='black', alpha=0.2, linewidth=0.5)
    plt.plot(x.T, z.T, color='black', alpha=0.2, linewidth=0.5)


def plot_impl(grid, func,enhanceFactor=1.0,fig=1,clear=True):
    dim = len(grid.upper_bound)
    xyz = [0] * 3
    lb = grid.lower_bound
    ub = grid.upper_bound
    nc = np.array(grid.number_of_cells)* enhanceFactor
    
    for i in range(dim):
        xyz[i] = np.linspace(lb[i], ub[i], int(nc[i])+1)

    if dim > 1: 
        # Create grid variables matching the names used inside your 'func' string
        x, z = np.meshgrid(xyz[0], xyz[1], indexing='ij')
        
        # Evaluate the implicit function string safely using local variables
        volume = eval(func)
        
        contours = measure.find_contours(volume, level=0.0)
        if len(contours) > 0:
            contour = contours[0]
            
            # Map index grid back to actual bounding box coordinates
            # Note: indexing='ij' means contour[:, 0] maps to axis 0 (x) and contour[:, 1] maps to axis 1 (z)
            scaled_x = np.interp(contour[:, 0], [0, nc[0]-1], [lb[0], ub[0]])
            scaled_z = np.interp(contour[:, 1], [0, nc[1]-1], [lb[1], ub[1]])
            
            plt.figure(fig,clear=clear)
            plot_grid(x,z)
            plt.plot(scaled_x, scaled_z, color='k', linewidth=1.5, label='Outline')

        
    elif dim > 2:
        x, y, z = np.meshgrid(xyz[0], xyz[1],xyz[2], indexing='ij')
        volume = eval(func)
    
    
        # 3. Extract isosurface using Marching Cubes
        verts, faces, normals, values = measure.marching_cubes(volume, level=0.0)
        
        # 4. Create the trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Optional: check if the mesh is valid
        print(mesh.is_watertight)

def plot_impl_trimesh(grid, func,fig=1,clear=True):
    dim = len(grid.upper_bound)
    xyz = [0] * 3
    lb = grid.lower_bound
    ub = grid.upper_bound
    nc = grid.number_of_cells
    
    for i in range(dim):
        xyz[i] = np.linspace(lb[i], ub[i], int(nc[i]))

    if dim > 1: 
        # Create grid variables matching the names used inside your 'func' string
        x, z = np.meshgrid(xyz[0], xyz[1], indexing='ij')
        
        # Evaluate the implicit function string safely using local variables
        volume = eval(func)
        
        contours = measure.find_contours(volume, level=0.0)
        if len(contours) > 0:
            contour = contours[0]
            
            # Map index grid back to actual bounding box coordinates
            # Note: indexing='ij' means contour[:, 0] maps to axis 0 (x) and contour[:, 1] maps to axis 1 (z)
            scaled_x = np.interp(contour[:, 0], [0, nc[0]-1], [lb[0], ub[0]])
            scaled_z = np.interp(contour[:, 1], [0, nc[1]-1], [lb[1], ub[1]])
            
            vertices_2d = np.column_stack((scaled_x, scaled_z))
        
            ## FIX: Form pairs of indices to close the loop (0->1, 1->2, ..., N-1->0)
            num_verts = len(vertices_2d)
            lines = []
            for i in range(num_verts):
                start_idx = i
                end_idx = (i + 1) % num_verts # Loops back to 0 at the end
                lines.append(trimesh.path.entities.Line([start_idx, end_idx]))

            # Initialize the 2D Path with closed entities
            path = trimesh.path.Path2D(entities=lines, vertices=vertices_2d)
            
            # This will now successfully compute the interior triangles
            tri_vertices, tri_faces = path.triangulate()
            
            # Check if triangles were successfully generated before plotting
            if len(tri_faces) == 0:
                print("Triangulation failed. Ensure the contour forms a clean closed loop.")
                return

            tri_x = tri_vertices[:, 0]
            tri_z = tri_vertices[:, 1]
            
            triangulation = mtri.Triangulation(tri_x, tri_z, tri_faces)
            # triangulation = mtri.Triangulation(scaled_x, scaled_z)
            
            plt.figure(fig,clear=clear)
            plot_grid(x,z)
            plt.triplot(triangulation, 'k-', linewidth=0.5)

        
    elif dim > 2:
        x, y, z = np.meshgrid(xyz[0], xyz[1],xyz[2], indexing='ij')
        volume = eval(func)
    
    
        # 3. Extract isosurface using Marching Cubes
        verts, faces, normals, values = measure.marching_cubes(volume, level=0.0)
        
        # 4. Create the trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Optional: check if the mesh is valid
        print(mesh.is_watertight)


def gen_impl_vtk(grid, func,enhanceFactor=1.0,filename="output_outline.vtk"):
    dim = len(grid.upper_bound)
    xyz = [0] * 3
    lb = grid.lower_bound
    ub = grid.upper_bound
    nc = np.array(grid.number_of_cells)* enhanceFactor
    
    for i in range(dim):
        xyz[i] = np.linspace(lb[i], ub[i], int(nc[i]))

    if dim > 1: 
        # Create grid variables matching the names used inside your 'func' string
        x, z = np.meshgrid(xyz[0], xyz[1], indexing='ij')
        
        # Evaluate the implicit function string safely using local variables
        volume = eval(func)
        contours = measure.find_contours(volume, level=0.0)
        if len(contours) > 0:
            contour = contours[0]
            
            # Map index grid back to actual bounding box coordinates
            # Note: indexing='ij' means contour[:, 0] maps to axis 0 (x) and contour[:, 1] maps to axis 1 (z)
            scaled_x = np.interp(contour[:, 0], [0, nc[0]-1], [lb[0], ub[0]])
            scaled_z = np.interp(contour[:, 1], [0, nc[1]-1], [lb[1], ub[1]])
            save_contour_to_vtk(scaled_x, scaled_z, filename=filename)
            # save_clean_contour_to_vtk(scaled_x, scaled_z, filename="dielectric_sink_1.vtk")
            
    elif dim > 2:
        x, y, z = np.meshgrid(xyz[0], xyz[1],xyz[2], indexing='ij')
        volume = eval(func)
    
    
        # 3. Extract isosurface using Marching Cubes
        verts, faces, normals, values = measure.marching_cubes(volume, level=0.0)
        
        # 4. Create the trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Optional: check if the mesh is valid
        print(mesh.is_watertight)

def save_contour_to_vtk(scaled_x, scaled_z, filename="output_outline.vtk"):
    num_pts = len(scaled_x)
    
    with open(filename, "w") as f:
        # Write VTK standard header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Shape Outline Data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write the point coordinates (X, Y=0, Z)
        f.write(f"POINTS {num_pts} float\n")
        for i in range(num_pts):
            f.write(f"{scaled_x[i]} 0.0 {scaled_z[i]}\n")
            

        # f.write(f"\nLINES 1 {num_pts + 1}\n")
        f.write(f"\nPOLYGONS 1 {num_pts + 1}\n")
        
        # Format: [number of points in line] [index0] [index1] ... [index0]
        indices = list(range(num_pts)) + [0]  # Append 0 to close the loop
        indices_str = " ".join(map(str, indices))
        f.write(f"{num_pts + 1} {indices_str}\n")
        
    print(f"Saved clean shape outline to {filename}")
    
def save_clean_contour_to_vtk(scaled_x, scaled_z, filename="output_perfect_outline.vtk"):
    num_pts = len(scaled_x)
    pts = np.column_stack((scaled_x, np.zeros(num_pts), scaled_z))
    
    # 1. Detect where the line jumps (where distance between points is large)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    
    # Set a threshold slightly larger than your grid spacing
    # If a gap is larger than this, it's an artificial crossing jump
    threshold = (max(scaled_x) - min(scaled_x)) / 10.0 
    jump_indices = np.where(diffs > threshold)[0]
    
    # Split the point indices into separate independent lines
    line_segments = []
    start_idx = 0
    for jump in jump_indices:
        line_segments.append(list(range(start_idx, jump + 1)))
        start_idx = jump + 1
    line_segments.append(list(range(start_idx, num_pts)))
    
    # Calculate the exact size needed for the VTK connectivity table header
    # Total integers = (num of lines) + (sum of points in all lines)
    total_connectivity_entries = len(line_segments) + num_pts

    # 2. Write the formatted VTK file
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Clean Multi-Line Shape Outline\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write coordinates
        f.write(f"POINTS {num_pts} float\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
            
        # Write separate lines to break the interior connection
        f.write(f"\nLINES {len(line_segments)} {total_connectivity_entries}\n")
        for segment in line_segments:
            segment_str = " ".join(map(str, segment))
            f.write(f"{len(segment)} {segment_str}\n")
            
    print(f"Saved clean wireframe mesh with {len(line_segments)} separated lines.")


