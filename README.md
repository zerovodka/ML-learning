# ML - Learning

This is a history repository for learning ML.

```
In this repo, all output results from the notebook are exposed.
```

![Zerovodka Machine Learning](resources/Zerovodka.png "Zerovodka Machine Learning")

## Directory Structure
``` python
ML-learning
├── .github
├── resources
└── src
    └── {week}
        ├── data              # Downloaded data
        │
        ├── solution          # Improve study.ipynb
        │                     # Check the top of Notebook
        │
        │
        └── ***-study.ipynb # Just study and analysis
```

## What I learned

| Week | Study(src/{week})                                                                                                                                                                 | Solution(src/{week}/solution)                                                                                                                                       |
| :--- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `1`  | [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week1/1.3%20MNIST-study.ipynb) | [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/tree/master/src/week1/solution/) |

## If you wanna use my conda env

Save my zerovodka-ml-env.yml file.
then,

```bash
# create my env
conda env create -f zerovodka-ml-env.yml

# activate
conda activate zerovodka-ml

# connet kernel through Jupyter
python -m ipykernel install --user --name=zerovodka-ml --display-name "Python (zerovodka-ml)"

```

## Learning Environment

- Anaconda (conda env)
- Jupyter
- Github
- VSCode

```bash
# create conda env
conda create -n venv-name python=3.10

# activate conda env
conda activate venv-name

# install required packages
conda install jupyter etc...
# or
pip install jupyter etc...

# register conda env as a Jupyter Kernel
pip install ipykernel

# add kernel
python -m ipykernel install --user --name=venv-name --display-name "Python (venv-name)"
```

## Setting up Anaconda prompt in VSCode

```bash
# Find path of Anaconda
C:\Users\{Your Name}\anaconda3\Scripts\activate

# Anaconda execute this batch file
C:\Users\{Your Name}\anaconda3\condabin\conda.bat
```

Then, add following code to your VSCode Settings

```
Ctrl + Shift + P → "Preferences: Open Settings (JSON)"
```

```json
"terminal.integrated.profiles.windows": {
  "Anaconda Prompt": {
    "path": [
      "C:\\Windows\\System32\\cmd.exe"
    ],
    "args": ["/K", "C:\\Users\\{Your Name}\\anaconda3\\Scripts\\activate.bat"]
  }
},
```

## Collection of useful commands in Anaconda Prompt

| Command                          | Role                               | Description                 |
| :------------------------------- | :--------------------------------- | :-------------------------- |
| `cls`                            | clear                              | clear prompt                |
| `conda activate {env-name}`      | activate conda env                 | (base) → (conda env)        |
| `conda deactivate`               | deactivate conda env               | (conda env) → (base)        |
| `conda info --envs`              | show all created conda env         | also check current env      |
| `conda list`                     | all packages installed in this env | more detailed than pip list |
| `conda install {package name}`   | install packages                   | ex: `conda install pandas`  |
| `conda update {package name}`    | update packages                    |                             |
| `conda remove {package name}`    | remove packages                    |                             |
| `conda env remove -n {env-name}` | remove conda env                   | very carefully              |
| `where python`                   | check Python path in this env      | want to check accurate path |
| `jupyter notebook`               | execute Jupyter                    |                             |
| `exit`                           | exit terminal                      | exit prompt                 |
