# MHD_GDDC_GPU
MHD_GDDC_GPU is a GPU accelerated code based on the TENO and Gradient Descent Divergence Cleaning method ( GDDC ) to solve the ideal magnetohydrodynamics equations.
# Compiling

MHD_GDDC_GPU requires the NVFORTRAN compiler ( download from: https://developer.nvidia.com/hpc-sdk-downloads ). The compilation script has been prepared (./source/make_mhd), and the compilation command is:

```
sh make_mhd
```

# Running

MHD_GDDC_GPU can only run on hardware equipped with NVIDIA GPUs. In addition one file have to specified:

* `input.dat`: A file that defines the physical and numerical setup of the simulation, which can be customized based on specific requirements. Examples of input.dat files can be found in the `cases` folder.

Thus, to run a simulation, type, e.g.:
```
./MHD_GDDC_GPU.out ./input.dat
```

# Interpreting the `input.dat` file

`casename` defines the case name

`xmin` defines the x-min boundary of the domain

`xmax` defines the x-max boundary of the domain

`ymin` defines the y-min boundary of the domain

`ymax` defines the y-max boundary of the domain

`dt` defines the time step

`End time` defines the total simulation time

`NTF` defines the output interval of results

`eta` defines the parameter $\eta$ of GDDC

`gamma` defines the parameter $\gamma$

`bxmin` defines the type of x-min boundary ( inlet -> inlet, out -> outflow, cyc -> cyclic, sym -> symmetry )

`bxmax` defines the type of x-max boundary ( inlet -> inlet, out -> outflow, cyc -> cyclic, sym -> symmetry )

`bymin` defines the type of y-min boundary ( inlet -> inlet, out -> outflow, cyc -> cyclic, sym -> symmetry )

`bymax` defines the type of y-max boundary ( inlet -> inlet, out -> outflow, cyc -> cyclic, sym -> symmetry )

`Inlet origin variables: rho, u, v, w, p, Bx, By, Bz` define the inlet boundary original variables

`restart` defines whether to restart a simulation ( 0 -> without restart, 1 -> restart )

# Interpreting the outputs

Once the calculation is started, a folder will be created in the current directory with the name format "case name+time." As the calculation progresses, MHD_GDDC_GPU will output three types of files in the folder: Info.dat, restart.*.dat, and TEC.*.dat.

* `Info.dat`: the file contains basic configuration information for the simulation.

* `restart.*.dat`: files are used to restart the simulation

* `TEC.*.dat`: result files that can be imported into Tecplot to post-processed.

# Contributing
We welcome any contributions and feedback that can help improve MHD_GDDC_GPU. If you would like to contribute to the tool, please contact the maintainers or open an Issue in the repository or a thread in Discussions. Pull Requests are encouraged, but please propose or discuss the changes in the associated Issue beforehand.

# Licencing
Please refer to the licence file.

