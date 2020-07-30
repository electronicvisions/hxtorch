# hxtorch: PyTorch for BrainScaleS-2

The software in this repository has been developed by staff and students
of Heidelberg University as part of the research carried out by the
Electronic Vision(s) group at the Kirchhoff-Institute for Physics.
The research is funded by Heidelberg University, the State of
Baden-Württemberg, the European Union Sixth Framework Programme no.
15879 (FACETS), the Seventh Framework Programme under grant agreements
no 604102 (HBP), 269921 (BrainScaleS), 243914 (Brain-i-Nets), the
Horizon 2020 Framework Programme under grant agreement 720270, 785907, 945539 (HBP) as
well as from the Manfred Stärk Foundation.

This repository contains

* hxtorch -- a PyTorch extension for BrainScaleS-2

## How to build
### Build- and runtime dependencies
All build- and runtime dependencies are encapsulated in a [Singularity Container](https://sylabs.io/docs/).
If you want to build this project outside the Electronic Vision(s) cluster, please start by downloading the most recent version from [here](https://openproject.bioai.eu/containers/).

For all following steps, we assume that the most recent Singularity container is located at `/containers/stable/latest` – you are free to choose any other path.

### Github-based build
To build this project from public resources, adhere to the following guide:

```shell
# 1) Most of the following steps will be executed within a singularity container
#    To keep the steps clutter-free, we start by defining an alias
shopt -s expand_aliases
alias c="singularity exec --app dls /containers/stable/latest"

# 2) Prepare a fresh workspace and change directory into it
mkdir workspace && cd workspace

# 3) Fetch a current copy of the symwaf2ic build tool
git clone https://github.com/electronicvisions/waf -b symwaf2ic symwaf2ic

# 4) Build symwaf2ic
c make -C symwaf2ic
ln -s symwaf2ic/waf

# 5) Setup your workspace and clone all dependencies (--clone-depth=1 to skip history)
c ./waf setup --repo-db-url=https://github.com/electronicvisions/projects --project=hxtorch

# 6) Build the project
#    Adjust -j1 to your own needs, beware that high parallelism will increase memory consumption!
c ./waf configure
c ./waf build -j1

# 7) Install the project to ./bin and ./lib
c ./waf install

# 8) If you run programs outside waf, you'll need to add ./lib and ./bin to your path specifications
export SINGULARITYENV_PREPEND_PATH=`pwd`/bin:$SINGULARITYENV_PREPEND_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=`pwd`/lib:$SINGULARITYENV_LD_LIBRARY_PATH
export PYTHONPATH=`pwd`/lib:$PYTHONPATH
```

### On the Electronic Vision(s) Cluster

* Work on the frontend machine, `helvetica`. You should have received instructions how to connect to it.
* Follow [aforementioned instructions](#github-based-build) with the following simplifications
  * Replace **steps 3) and 4)** by `module load waf`
  * Make sure to run **step 6)** within a respective slurm allocation: Prefix `srun -p compile -c8`; depending on your shell, you might need to roll out the `c`-alias.
  * Replace **step 8)** by `module load localdir`.

## License
```
hxtorch: PyTorch for BrainScaleS-2 ('hxtorch')
Copyright (C) 2019–2020 Electronic Vision(s) Group
                        Kirchhoff-Institute for Physics
                        Ruprecht-Karls-Universität Heidelberg
                        69120 Heidelberg
                        Germany

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
USA
```
