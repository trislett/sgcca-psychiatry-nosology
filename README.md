The package sparsemodels (which wraps functions and adds parallel computing from the [RGCCA R packaged](https://github.com/rgcca-factory/RGCCA)) is used to calculate the multi-view sparse generalized (SGCCA) model.

The details and source code for sparsemodels are available here: https://github.com/trislett/sparsemodels.

Further information about functions are available using the python help. E.g., help(parallel_sgcca)
The sparsemodels package has been tested on M1 Mac (Sonoma 14.4.1), Ubuntu 22.04, and using Arch linux. 

Install sparsemodels using:

`pip install -U git+https://github.com/trislett/sparsemodels`

Please consult the [setup.py](https://github.com/trislett/sparsemodels/blob/main/setup.py) for a list of dependencies. Note, the bleeding edge of numpy may cause issues
with building the python package. If you experience this, please downgrade to numpy=1.23.4.

The annotated script of commands for calculting the SGCCA model are in [run_sgcca.py](https://github.com/trislett/sgcca-psychiatry-nosology/blob/main/run_sgcca.py).
