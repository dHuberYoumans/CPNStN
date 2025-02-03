# CPNStN

##### Table of Contents
- [Getting Started](#getting-started)  
- [Prerequisites](#prerequisites)
- [Running main.py](#running-mainpy)

## Getting Started

### GNU Screen

1. Download the source and extract 
```
$ wget http://git.savannah.gnu.org/cgit/screen.git/snapshot/v.4.3.1.tar.gz
$ tar -xvf v.4.3.1.tar.gz
$ cd v.4.3.1/src/
```

2. Build GNU Screen
```
$ ./autogen.sh
$ ./configure
$ make
```

3. Run GNU Screen
``` 
$ ./screen -S <session_name>
```

We recommend to create an alias for `v.4.3.1/src/screen`.

## Prerequisites

Dependencies are listed in [environment.yml](https://github.com/dHuberYoumans/CPNStN/blob/main/environment.yml) which can be used with anaconda to create the virtual environment _cpn_:
```
$ conda env create -f environment.yml
```

## Running main.py

### Locally
The main script uses torch's _ Distributed Data Parallel_ and has to be called using `torchrun`.

1. navigate to `CPNStN/lattice`
2. exectue `main.py` using `torchrun`

```
$ torchrun --nnodes=1 --nproc_per_node=2 main.py
```


### On Tursa
When running the scripts in an interactive session at Tursa, for conectivity purposes, we recommend to use GNU Screen. 
After allocating resrouces using `salloc`, 

1. activate the environment _cpn_
2. navigate to `CPNStN/lattice/`
3. use `srun` to execute `torchrun`

**Example**:
```
$ salloc -N1 --time=00:10:00 --qos=dev --partition=gpu
$ conda activate cpn
$ cd CPNStN/lattice/
$ srun torchrun --nnodes=1 --nproc_per_node=4 main.py
```

