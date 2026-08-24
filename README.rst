
WarpX Utilities (wxutils)
=========================

A collection of python functions to aid in `WarpX`_ python model development. This contains extra features, diagnostics, and helper functions for `PICMI`_ input files. 

All features and diagnostics are hardware agnostic unless otherwise stated.

Availiable Features
-------------------

- secondary emission model
- dielectric surface charge deposition

Availiable Diagnostics
----------------------

- 1D history diagnostic
   
  - Logs scalar data and periodically stores to disc
  - Stores data in h5 VizSchema format which can be viewed easily from `VisIt`_ using curve 

- 2D/3D field diagnostic [#f1]_
  
  - Saves user generated MultiFab fields in `npy` and `h5` format.
  - stores directly from memory, no unnecesary memory copies
  
.. [#f1] These features are under development.

Install
-------

create a Python environment using whatever method suits you.

PyPi
^^^^

Ensure warpx is installed to an environment::

     pip install warpx


Clone the package::
     
     git clone git@github.com:pearlmam/wxutils.git
     cd wxutils
     
Install using an editable install::
     
     pip install -e .
     
Conda
^^^^^

It is generally advised not to mix conda and PyPi packages; use either all one or all the other. Because of this, at this point, make sure your environment has everything it needs manually. Take a look at the toml file to see what packages are required.

install hatchling::

     conda install -c conda-forge hatchling
     
Install using an editable install::   

     pip install --no-deps --no-build-isolation -e .
     
This last step uses pip, but its the only way to install an editable install?



.. _WarpX: https://github.com/BLAST-WarpX/warpx

.. _PICMI: https://warpx.readthedocs.io/en/latest/usage/python.html
     
.. _VisIt: https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.1.2/index.html#



   
