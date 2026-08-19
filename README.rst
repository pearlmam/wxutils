
WarpX Utilities (wxutils)
=========================

A collection of python functions to aid in model development. This contains callback methods, grid and geometry managment methods, and more....

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

     pip install --no-build-isolation -e .
     
This last step uses pip, but its the only way to install an editable install?



     
     




   
