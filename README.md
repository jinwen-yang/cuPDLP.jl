# cuPDLP.jl

This repository contains experimental code for solving linear programming using first-order methods on NVIDIA GPUs. 

Part of the code utilizes https://github.com/google-research/FirstOrderLp.jl which originally stated Apache-2.0 as its license.

## Setup

A one-time step is required to set up the necessary packages on the local machine:

```shell
$ julia --project -e 'import Pkg; Pkg.instantiate()'
```

## Running 

`solve.jl` is the recommended script for using cuPDLP. The results are written to JSON and text files. All commands below assume that the current directory is the working directory.

```shell
$ julia --project scripts/solve.jl \
    --instance_path=INSTANCE_PATH \
    --output_directory=OUTPUT_DIRECTORY \ 
    --tolerance=TOLERANCE \
    --time_sec_limit=TIME_SEC_LIMIT
```

## Interpreting the output

A table of iteration stats will be printed with the following headings.

##### runtime

`#iter` = the current iteration number.

`#kkt` = the cumulative number of times the KKT matrix is multiplied.

`seconds` = the cumulative solve time in seconds.

##### residuals

`pr norm` = the Euclidean norm of primal residuals (i.e., the constraint
violation).

`du norm` = the Euclidean norm of the dual residuals.

`gap` = the gap between the primal and dual objective.

##### solution information

`pr obj` = the primal objective value.

`pr norm` = the Euclidean norm of the primal variable vector.

`du norm` = the Euclidean norm of the dual variable vector.

##### relative residuals

`rel pr` = the Euclidean norm of the primal residuals, relative to the
right-hand-side.

`rel dul` = the Euclidean norm of the dual residuals, relative to the primal
linear objective.

`rel gap` = the relative optimality gap.