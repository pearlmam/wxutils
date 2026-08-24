
from pywarpx import picmi
constants = picmi.constants

availiable_species = {
    'electron': {'mass': constants.m_e, "charge": -constants.q_e}
}

boundaryEnable = {'x_lo': 'save_particles_at_xlo',
                  'x_hi': 'save_particles_at_xhi',
                  'y_lo': 'save_particles_at_ylo',
                  'y_hi': 'save_particles_at_yhi',
                  'z_lo': 'save_particles_at_zlo',
                  'z_hi': 'save_particles_at_zhi',
                  'eb': 'save_particles_at_eb'
                  }


def set_species_params(species, boundary=None):
    if (species.particle_type is None):
        pass
    elif (species.particle_type in availiable_species) and (species.mass is None) or (species.charge is None):
        species.mass = availiable_species[species.particle_type]["mass"]
        species.charge = availiable_species[species.particle_type]["charge"]
        species.particle_type = None  # this still generates warning...
    else:
        raise Exception("Particle type '%s' not supported for set_species_params(), define mass and charge manually.")

    if not isinstance(boundary, (str, list, tuple, type(None))):
        raise TypeError("boundary must be of type string, list, tuple, or None")

    if isinstance(boundary, (str)):
        boundary = [boundary]
    if isinstance(boundary, (list, tuple)):
        for _boundary in boundary:
            setattr(species, boundaryEnable[_boundary], True)
    return species



def get_valid_region(arr, ngrow):
    """Slices an array view down to its valid interior, stripping guard cells."""
    slices = []
    for ng in ngrow:
        if ng == 0:
            slices.append(slice(None))
        else:
            slices.append(slice(ng, -ng))
    return arr[tuple(slices)]