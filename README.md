# CPNStN

##### Table of Contents
- [Getting Started](#getting-started)  
- [Prerequisites](#prerequisites)
- [Running main.py](#running-mainpy)

## Getting Started

### File Tree

#### toy_model/
```
.
├── data
│   ├── samples_n2_b1.0.dat
│   ├── samples_n2_b1.0_mII.dat
│   └── ...
├── main.py
├── plots
│   ├── fuzzy-one
│   │   └── 2025.02.15_17:42
│   │       ├── deformation_params.pdf
│   │       ├── errorbars_comp.pdf
│   │       ├── loss.pdf
│   │       ├── raw_data
│   │       └── run.log
│   ├── one-pt
│   │   └── 2025.02.15_17:31
│   │       └── ...
│   └── two-pt
│       └── 2025.02.15_17:35
│           └──...
└── src
    ├── analysis.py
    ├── deformations.py
    ├── linalg.py
    ├── losses.py
    ├── mcmc.py
    ├── model.py
    ├── observables.py
    └── utils.py
```

#### lattice/
```
.
├── data
│   ├── cpn_b4.0_L64_Nc3_ens.dat
│   └── cpn_b4.0_L64_Nc3_u.dat
├── main.py
├── plots
│   ├── one-pt
│   │   └── 2025.02.12_17:15
│   │       ├── deformation_params.pdf
│   │       ├── deformation_params_norms.pdf
│   │       ├── errorbars_comp.pdf
│   │       ├── loss.pdf
│   │       ├── raw_data
│   │       │   ├── af.pt
│   │       │   ├── losses_train.pt
│   │       │   ├── losses_val.pt
│   │       │   ├── model.pt
│   │       │   └── observable.pt
│   │       └── run.log
│   └── two-pt
│       └── 2025.02.14_19:14
│           └── ...
└── src
    ├── analysis.py
    ├── deformations.py
    ├── linalg.py
    ├── losses.py
    ├── model.py
    ├── observables.py
    ├── unet.py
    └── utils.py
```

### Example run.log
```
2025-02-14 19:14:16,341 - INFO: Used Parameters

+---------------------+------------------+
| param               | value            |
+=====================+==================+
| device              | cuda:0           |
+---------------------+------------------+
| L (lattice size)    | 64               |
+---------------------+------------------+
| beta (coupling cst) | 4.0              |
+---------------------+------------------+
| n (dimC CP)         | 2                |
+---------------------+------------------+
| dim_g               | 8                |
+---------------------+------------------+
| lr (learning rate)  | 1e-05            |
+---------------------+------------------+
| batch size          | 256              |
+---------------------+------------------+
| loss_fn             | rloss            |
+---------------------+------------------+
| epochs              | 10000            |
+---------------------+------------------+
| obs                 | LatTwoPt         |
+---------------------+------------------+
| (p,q)               | ((8, 8), (8, 9)) |
+---------------------+------------------+
| (i,j)               | (0, 1)           |
+---------------------+------------------+
| (k,l)               | (0, 1)           |
+---------------------+------------------+
| SLURM_JOB_ID        | 48886            |
+---------------------+------------------+
```


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

### Usage main.py
usage: main.py [-h] [--obs OBS] [--i I] [--j J] [--particle PARTICLE] --tag TAG [--deformation DEFORMATION] [--epochs EPOCHS] [--loss_fn LOSS_FN] [--batch_size BATCH_SIZE] [--load_samples LOAD_SAMPLES]

options:
  -h, --help            show this help message and exit
  --obs OBS             observable: ToyOnePt | ToyTwoPt
  --i I                 z component
  --j J                 z* component
  --particle PARTICLE   0 => z, 1 => w
  --tag TAG             tag for saving (one-pt | two-pt | fuzzy-one)
  --deformation DEFORMATION
                        type of deformation: Linear | Homogeneous
  --epochs EPOCHS       epochs
  --loss_fn LOSS_FN     loss function
  --batch_size BATCH_SIZE
                        batch size
  --load_samples LOAD_SAMPLES
                        which samples to load, those created sequentuially (seq) or with parallel (II) metropolis updates

### Locally 
The (toy_model) main script uses torch's _ Distributed Data Parallel_ and has to be called using `torchrun`.

1. navigate to `CPNStN/lattice`
2. exectue `main.py` using `torchrun`

```
$ torchrun --nnodes=1 --nproc_per_node=2 main.py \
    --obs=ToyFuzzyOne \
    --i=0 \
    --j=1 \
    --tag=fuzzy-one \
    --epochs=1000 \
    --deformation=Homogeneous \
    --loss_fn=rlogloss \
    --batch_size=1024
```

### On Tursa
#### Interactive Session

When running the scripts in an interactive session at Tursa, for conectivity purposes, we recommend to use GNU Screen. 
After allocating resrouces using `salloc`, 

1. activate the environment _cpn_
2. navigate to `CPNStN/lattice/`
3. use `srun` to execute `torchrun`

**Example srun**:
```
$ salloc -N1 --time=00:10:00 --qos=dev --partition=gpu
$ conda activate cpn
$ cd CPNStN/lattice/
$ srun torchrun --nnodes=1 --nproc_per_node=4 main.py --obs=LatTwoPt --p="(5,7)" --q="(11,13)" --i=0 --j=1 --k=0 --ell=1 --tag=two-pt --epochs=1000 --batch_size=128
```

#### Using Slurm
**Example slurm job script**:
```
#!/bin/bash

# Slurm job options
#SBATCH --job-name=cpn_lat_uet
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=[NAME]

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

#SBATCH --output="slurm_logs/two-pt/slurm-%j.out"

# load modules
module load gcc/9.3.0
module load cuda/12.3
module load openmpi/4.1.5-cuda12.3

source activate
conda activate pytorch2.5

cd ~/CPNStN/lattice

# name of script
application="main.py"


# run script
echo 'working dir: ' $(pwd)
echo $'\nrun started on ' `date` $'\n'

export OMP_NUM_THREADS=4

srun torchrun \
	--nnodes=1 \
	--nproc_per_node=4 \
	${application} \
	--obs=LatTwoPt \
	--p="(8,8)" \
	--q="(8,9)" \
	--i=0 \
	--j=1 \
	--k=0 \
	--ell=1 \
	--tag=two-pt \
	--loss_fn=rloss \
	--epochs=10000 \
	--batch_size=256
 
echo $'\nrun completed on ' `date`
```


