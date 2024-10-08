[![DOI](https://zenodo.org/badge/796674797.svg)](https://zenodo.org/doi/10.5281/zenodo.12167725)

The package sparsemodels (which wraps functions and adds parallel computing from the [RGCCA R packaged](https://github.com/rgcca-factory/RGCCA)) is used to calculate the multi-view sparse generalized (SGCCA) model.

The details and source code for sparsemodels are available here: https://github.com/trislett/sparsemodels.

Further information about functions are available using the python help. E.g., help(parallel_sgcca).

The sparsemodels package has been tested on M1 Mac (Sonoma 14.4.1), Ubuntu 22.04, Ubuntu 24.04 and using Arch linux. 

Lazy install using Conda:

	git clone https://github.com/trislett/sgcca-psychiatry-nosology
	cd sgcca-psychiatry-nosology
	conda env create -f environment.yml
	conda activate sparsemodels_env
	pip install git+https://github.com/trislett/sparsemodels

Test if sparse models is working:

	./simulate_sgcca.py

Alternatively, just install sparsemodels using the python environment of your choice:

`pip install git+https://github.com/trislett/sparsemodels`

Please consult the [setup.py](https://github.com/trislett/sparsemodels/blob/main/setup.py) for a list of dependencies. 

The annotated script of commands for calculting the SGCCA model are in [run_sgcca.py](https://github.com/trislett/sgcca-psychiatry-nosology/blob/main/run_sgcca.py).

An example using random data is available in [simulate_sgcca.py](https://github.com/trislett/sgcca-psychiatry-nosology/blob/main/simulate_sgcca.py).
