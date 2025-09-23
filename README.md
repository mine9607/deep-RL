# Install Dependencies

Note there is a gpu enabled yml file and a cpu only yml file for non-accelerated machines

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
