# ESSOS

by Estêvão Gomes ([@EstevaoMGomes](https://github.com/EstevaoMGomes))

This project is a Stellarator Coil Optimizer of alpha particles via differentiable JAX code and was developed as the research 
work for the New Talents in Physics Fellowship, awarded by the [Calouste Gulbenkian Foundation](https://gulbenkian.pt/en/).

The project was developed under the supervision of professor Rogério Jorge ([@rogeriojorge](https://github.com/rogeriojorge)).

## Repository Organization

## Abstract
In magnetic confinement fusion, one of the most promising approaches that allows steady state
operation with no disruptions is the stellarator. A stellarator consists of electromagnetic coils
that create a twisted magnetic field that needs to be optimized to confine a high-performing
plasma. Such optimization is performed over a large set of parameters, typically of the order of
several hundred or more. Furthermore, the target magnetic field is usually a fixed one, that has
been previously obtained using another optimization based on the ideal MHD equations. With
this work, we trace particles directly in the corresponding Biot-Savart magnetic fields stemming
from a set of coils and optimize them to yield a small fraction of lost particles outside of the
confinement region. Furthermore, to replace the need for hundreds of simulations per
optimization step, we make use of automatic differentiation by implementing the guiding-center
equations, magnetic field solver, and optimization routines in JAX. This allows us to streamline
optimization efforts, and create a specialized, but very fast, numerical tool, to optimize force-free
stellarator equilibria. As force-free equilibria are usually the first step in determining the
viability of a device, such optimizations will be able to guide future designs based on ideal
MHD equilibria.

## How to use the repository
After cloning the repository:
```
git clone https://github.com/uwplasma/ESSOS.git
```
The easiest way to run an example script is to create a conda environment.
To install conda run on the terminal:

```
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
```
Or install any other version avalaible on [https://repo.anaconda.com/archive/](https://repo.anaconda.com/archive/).
Then Run:
```
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ~/anaconda
rm Anaconda3-2024.06-1-Linux-x86_64.sh
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 
```
To refresh, run 
```
source .bashrc
```
Then you can create a conda environment with
```
conda env create -f depenencies_file.yml
```
where "dependencies_file.yml" can be dependencies_gpu.yml or dependencies_cpu.yml, whether you want to run the scripts on CPU or GPU.


## Without setting an environment,the main pip packages required are: 
```
pip install matplotlib
pip install simsopt
```
For GPU enable jax use this (it requires cuda12 installation, with the corresponding nvcc, see JAX documentation for more details on this):
```
pip install -U "jax[cuda12]"==0.4.43
```
For CPU only usage use instead: 
```
pip install -U jax==0.4.43
```
The following packages need the JAX package above
```
pip install equinox==0.11.9
pip install lineax==0.0.6
pip install optimistix==0.0.8
pip install diffrax==0.6.0
```
## Tracing particles from coils
An example script for tracing particles in a magnetic field generated from coils obtained
from a 
```
bio_savart_opt.json 
```
file of SIMSOPT can be found in the path 
```
ESSOS/examples/trace_particles_from_coils.py 
```
This can be run using the command: 
```
python trace_particles_from_coils.py 
```
More details on how to change the different aprameters can be seen inside the script 
```
trace_particles_from_coils.py 
```
The bio_savart_opt.json can be found inside the folder ESSOS/examples/inputs and represents a coils set obtained from 
a SIMSOPT two-stage coils optimisation using the VMEC configuration represented by the file
```
ESSOS/examples/inputs/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc 
```