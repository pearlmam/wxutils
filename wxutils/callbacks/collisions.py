# -*- coding: utf-8 -*-

from pywarpx import libwarpx
from pywarpx.LoadThirdParty import load_cupy
from wxutils.grid import get_grid_cell_sizes

INVALID_ID = 16777216 # 2**24, validity bit in the idcpu (particle cpu id), signals a kill to the particles
neutral_accumulator = {}
def apply_recombination_decay(sim,p0='electrons',p1='he_ions',alpha=1.0e-13):
    """
    Callback function that runs every timestep. 
    It decays electrons & ions equally per tile and accumulates neutral weight.
    """
    xp,_ = load_cupy()

    p0_pc = sim.particles.get(p0)
    p1_pc = sim.particles.get(p1)
    dt = libwarpx.warpx.getdt(0)
    dx_min = min(get_grid_cell_sizes())
    # print(p0_pc)
    level=0
    p0_dict = p0_pc.get_particles(level)
    p1_dict = p1_pc.get_particles(level)
    
    p0_wi = p0_pc.get_real_comp_index('w')
    p1_wi = p1_pc.get_real_comp_index('w')
    
    pos_indices = []
    for dim in ['x', 'y', 'z']:
        try:
            idx = p0_pc.get_real_comp_index(dim)
            pos_indices.append(idx)
        except:
            # If the dimension doesn't exist (e.g., 'y' in a 1D or 2D XZ sim), 
            # WarpX will throw an error; we just skip it.
            continue
    
    # Loop over every parallelized tile of our grid
    for (tile_idx,p0_tile), (_,p1_tile) in zip(p0_dict.items(),p1_dict.items()):
        
        
        p0_soa = p0_tile.get_struct_of_arrays()  
        p1_soa = p1_tile.get_struct_of_arrays()
        
        p0_w = xp.asarray(p0_soa.get_real_data(p0_wi), copy=False)
        p1_w = xp.asarray(p1_soa.get_real_data(p1_wi), copy=False)
        #print(p1_w)
        # Ensure we have particles of both species in this block
        if len(p0_w) < 2 or len(p1_w) < 2:
            continue
        
        tile_vol = 1.0
        for idx in pos_indices:
            coords = xp.asarray(p1_soa.get_real_data(idx), copy=False)
            
            # Calculate spread for this specific dimension
            spread = xp.max(coords) - xp.min(coords)
            
            # Apply safety floor
            spread = xp.maximum(spread, dx_min)
            
            # Multiply into the total hyper-volume
            tile_vol *= spread
        
        # --- SAFETY GUARD 2: Prevent volume from shrinking below a physical cell size ---
        # dx = 0.00156 from your simulation setup

        # Total local macroparticle physical weight (sum of physical charges)
        total_e_physical = xp.sum(p0_w)
        n_e = total_e_physical / tile_vol # average local density (m^-3)
        
        # Probability that a macroparticle recombines in this time-step
        # P = alpha * n_e * dt
        prob_decay = min(alpha * n_e * dt, 1.0)
        
        if prob_decay <= 0:
            continue
            
        # --- Decay Implementation (Altering Weights) ---
        # Calculate how much total macroparticle weight we need to decay
        decayed_weight_p0 = p0_w * prob_decay
        decayed_weight_p1 = p1_w * prob_decay
        
        # Update the kinetic weights in memory (directly writeable)
        p0_w -= decayed_weight_p0
        p1_w -= decayed_weight_p1
        
        # Zero out extremely low-weight "ghost" particles to save computation
        p0_mask = p0_w < 1e-10
        p1_mask = p1_w < 1e-10
        
        p0_id = xp.array(p0_soa.get_idcpu_data(), copy=False) # This is the internal AMReX ID/CPU buffer
        p1_id = xp.array(p1_soa.get_idcpu_data(), copy=False)
        # print(p0_id)
        p0_id[p0_mask] = INVALID_ID
        p1_id[p1_mask] = INVALID_ID
        
        p0_pc.redistribute() 
        p1_pc.redistribute()
        
        #print(p0_id)
        # --- Accumulating the Background Neutral Gas ---
        # We average the deleted weight of electrons & ions to ensure exact mass/charge conservation
        neutral_weight_created = 0.5 * (xp.sum(decayed_weight_p0) + xp.sum(decayed_weight_p1))
        
        if tile_idx not in neutral_accumulator:
            neutral_accumulator[tile_idx] = 0.0
        neutral_accumulator[tile_idx] += neutral_weight_created
        
        
    #print(neutral_accumulator)
