# Install Dependencies

> conda export -> creates an overly verbose .yml file and should not be used for install. Also, a simplified env.yml file doesn't work with both conda and pip dependencies. The correct solution should be to create a separate dependency file for conda and pip. First create the conda env from the conda file and then activate the environment before running the pip install script.

## Option 1 - Install Everything (env.yml)

```bash
conda env create -f environment.yml -n rl_gpu
```

## Option 2 - Conda only libraries

```bash
conda env create -f conda_only.yml -n rl_gpu
```

## Option 3 - Pip only libraries

```bash
conda create -n rl_gpu
conda activate rl_gpu
pip install -r pip_requirements.txt
```
