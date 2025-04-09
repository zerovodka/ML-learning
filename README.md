# ML - Learning

This is a history repository for learning ML.

```
해당 레포지토리에서는, 모든 출력 값이 남아있습니다.

In this repo, all output results from the notebook are exposed.
```

## README 바로 가기

```
각 프로젝트에 대한 전반적인 내용과 트러블슈팅 내용이 적혀있습니다
```

| 주차 | README 링크                                                                                                                                         |
| :--- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`  |                                                                                                                                                     |
| `2`  |                                                                                                                                                     |
| `3`  | [![README](https://img.shields.io/badge/Github-WEEK__3-black?logo=github)](https://github.com/zerovodka/ML-learning/blob/master/src/week3/WEEK3.md) |

## Stacks

- Python
- PyTorch
- Hugging Face Transformers & Datasets
- Google Colab / CUDA
- Matplotlib

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
        │   │                 # 실습 코드 기반 개선 및 프로젝트 수행
        │   └── ***-solution-***.ipynb
        │               .
        │               .
        │               .
        │
        │
        ├── WEEK{주차}      # 해당 주차를 전반적으로 볼 수 있는 README
        │
        └── ***-study.ipynb # Just study and analysis / 실습 및 공부
```

## What I learned

| Week | Study(src/{week})                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Solution(src/{week}/solution)                                                                                                                                                                                                                                                                                                                                                                                                                             | 주요 학습 내용                                                                                                                                                                                                                                                                                                                                                                                                                       |
| :--- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`  | `MLP`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week1/MNIST-study.ipynb?flush_cache=true)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `Basic Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week1/solution/MNIST-solution-basic.ipynb) <br> `Advanced Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week1/solution/CIFAR-solution-advanced.ipynb?flush_cache=true) | - `MNIST`<br>&emsp;- CrossEntropyLoss/MSE<br>&emsp;- Train/Test 정확도, 시각화 <br> - `CIFAR10`<br>&emsp;- Optimizer(SGD vs Adam 비교)<br>&emsp;- Activation Func (LeakyReLU vs Sigmoid)<br>&emsp;- Dropout generalization 성능 테스트                                                                                                                                                                                               |
| `2`  | `Transformer` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/Transformer-study.ipynb?flush_cache=true)<br>`BERT vs GPT2 Tokenizer` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/BERT-vs-GPT-Tokenizer.ipynb?flush_cache=true)<br>`Embedding` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/Embedding.ipynb?flush_cache=true)<br>`Positional Encoding` <br> [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/Positional-Encoding.ipynb?flush_cache=true) | `Basic Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/solution/Transformer-solution-basic.ipynb) <br>`Advanced Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week2/solution/Transformer-solution-advanced.ipynb)       | `IMDb` - Binary Classification / Last Word Prediction<br><br> -Tokenizer<br>&emsp;- BERT vs GPT2 Tokenizer(`emoji` side)<br>&emsp;- AutoTokenizer<br>- Embedding process<br>&emsp;-BERT vs SBERT similarity 비교 및 시각화<br>- Positional Encoding<br>- Transformer <br>&emsp;- ignore token filtering<br>&emsp;- Self-Attention<br>&emsp;- Feed Foward Layer<br>&emsp;- Loss, Accuracy 시각화<br>&emsp;- Multi-Head-Attention 구현 |
| `3`  | `Transfer Learning`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week3/DistilBERT-study.ipynb?flush_cache=true)<br>`AG_News`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week3/AG_News.ipynb?flush_cache=true)                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `Basic Level`<br>[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://nbviewer.org/github/zerovodka/ML-learning/blob/master/src/week3/solution/DistilBERT-solution-basic.ipynb?flush_cache=true) <br><br>`README 보러가기`<br>[![README](https://img.shields.io/badge/Github-WEEK__3-black?logo=github)](https://github.com/zerovodka/ML-learning/blob/master/src/week3/WEEK3.md)                                     | `IMDB` - Binary Classification<br>`AG_News` - Multi-Class Classification<br><br>- DistilBERT Model 학습<br>- AG_News 데이터셋 톺아보기<br>- Transfer Learning<br>&emsp;- fine tuning: `freeze` 기법<br>- Multi-Class Classification에 맞는 모델 및 정확도 함수 + 시각화                                                                                                                                                              |

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
