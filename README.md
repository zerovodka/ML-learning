# ML - Learning

This is a history repository for learning ML.

```
In this repo, all output results from the notebook are exposed.
```

![Zerovodka Machine Learning](resources/Zerovodka.png "Zerovodka Machine Learning")

## Directory Structure

```python
ML-learning
├── .github
├── resources
└── src
    └── {week}
        ├── data              # Downloaded data
        │
        ├── solution          # Improve study.ipynb
        │   │                 # Check the top of Notebook
        │   └── ***-solution-***.ipynb
        │               .
        │               .
        │               .
        │
        └── ***-study.ipynb # Just study and analysis
```

## What I learned

| Week | Study(src/{week})                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Solution(src/{week}/solution)                                                                                                                                                                                                                                                                                                                                                                                                                             | 비고                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :--- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`  | `MLP`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week1/MNIST-study.ipynb?flush_cache=true)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `Basic Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week1/solution/MNIST-solution-basic.ipynb) <br> `Advanced Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week1/solution/CIFAR-solution-advanced.ipynb?flush_cache=true) | - MNIST<br>&emsp;- CrossEntropyLoss/MSE<br>&emsp;- Train/Test 정확도, 시각화 <br> - CIFAR10<br>&emsp;- Optimizer(SGD vs Adam 비교)<br>&emsp;- Activation Func (LeakyReLU vs Sigmoid)<br>&emsp;- Dropout generalization 성능 테스트                                                                                                                                                                                                   |
| `2`  | `Transformer` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/Transformer-study.ipynb?flush_cache=true)<br>`BERT vs GPT2 Tokenizer` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/BERT-vs-GPT-Tokenizer.ipynb?flush_cache=true)<br>`Embedding` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/Embedding.ipynb?flush_cache=true)<br>`Positional Encoding` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/Positional-Encoding.ipynb?flush_cache=true) | `Basic Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/solution/Transformer-solution-basic.ipynb) <br>`Advanced Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/solution/Transformer-solution-advanced.ipynb)       | `IMDb` - Binary Classification / Last Word Prediction<br><br> -Tokenizer<br>&emsp;- BERT vs GPT2 Tokenizer(`emoji` side)<br>&emsp;- AutoTokenizer<br>- Embedding process<br>&emsp;-BERT vs SBERT similarity 비교 및 시각화<br>- Positional Encoding<br>- Transformer <br>&emsp;- ignore token filtering<br>&emsp;- Self-Attention<br>&emsp;- Feed Foward Layer<br>&emsp;- Loss, Accuracy 시각화<br>&emsp;- Multi-Head-Attention 구현 |
| `3`  | `Transfer Learning`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week3/DistilBERT-study.ipynb?flush_cache=true)<br>`AG_News`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week3/AG_News.ipynb?flush_cache=true)                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | ''                                                                                                                                                                                                                                                                                                                                                                                                                                                        | - Transfer Learning<br>&emsp;- DistilBERT binary classification fine tuning: `freeze` <br>- AG_News Dataset                                                                                                                                                                                                                                                                                                                          |

<!-- Colab 사용으로 인한 로컬 빌드 수요 없음에 따른 주석 처리 -->
<!-- ## If you wanna use my conda env

Save my zerovodka-ml-env.yml file.
then,

```bash
# create my env
conda env create -f zerovodka-ml-env.yml

# activate
conda activate zerovodka-ml

# connet kernel through Jupyter
python -m ipykernel install --user --name=zerovodka-ml --display-name "Python (zerovodka-ml)"

``` -->

## Learning Environment

- Anaconda (conda env)
- Jupyter
- Google Colab
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
