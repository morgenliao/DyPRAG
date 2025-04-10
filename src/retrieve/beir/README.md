<h1 align="center">
<img style="vertical-align:middle" width="450" height="180" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" />
</h1>

<p align="center">
    <a href="https://github.com/beir-cellar/beir/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/beir-cellar/beir.svg">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/beir-cellar/beir/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/beir-cellar/beir.svg?color=green">
    </a>
    <a href="https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://pepy.tech/project/beir">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/beir?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
    <a href="https://github.com/beir-cellar/beir/">
        <img alt="Downloads" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://openreview.net/forum?id=wCu6T5xFjeJ">Paper</a> |
        <a href="#beers-installation">Installation</a> |
        <a href="#beers-quick-example">Quick Example</a> |
        <a href="#beers-available-datasets">Datasets</a> |
        <a href="https://github.com/beir-cellar/beir/wiki">Wiki</a> |
        <a href="https://huggingface.co/BeIR">Hugging Face</a>
    <p>
</h4>

<!-- > The development of BEIR benchmark is supported by: -->

<h3 align="center">
    <a href="http://www.ukp.tu-darmstadt.de"><img style="float: left; padding: 2px 7px 2px 7px;" width="220" height="100" src="./images/ukp.png" /></a>
    <a href="https://www.tu-darmstadt.de/"><img style="float: middle; padding: 2px 7px 2px 7px;" width="250" height="90" src="./images/tu-darmstadt.png" /></a>
    <a href="https://uwaterloo.ca"><img style="float: right; padding: 2px 7px 2px 7px;" width="320" height="100" src="./images/uwaterloo.png" /></a>
</h3>

<h3 align="center">
    <a href="https://huggingface.co/"><img style="float: middle; padding: 2px 7px 2px 7px;" width="400" height="80" src="./images/HF.png" /></a>
</h3>

## :beers: 这是什么?

**BEIR** 是一个**异构基准**,包含多样化的信息检索任务。它还为在基准测试中评估基于NLP的检索模型提供了一个**通用且简单的框架**。

要**概览**,请查看我们的**新wiki**页面: [https://github.com/beir-cellar/beir/wiki](https://github.com/beir-cellar/beir/wiki)。

要了解**模型和数据集**,请查看我们的**Hugging Face (HF)**页面: [https://huggingface.co/BeIR](https://huggingface.co/BeIR)。

要了解**排行榜**,请查看**Eval AI**页面: [https://eval.ai/web/challenges/challenge-page/1897](https://eval.ai/web/challenges/challenge-page/1897)。

要了解更多信息,请查看我们的出版物:

- [BEIR: 用于信息检索模型零样本评估的异构基准](https://openreview.net/forum?id=wCu6T5xFjeJ) (NeurIPS 2021, 数据集与基准测试赛道)
- [酿造BEIR的资源：可复现的参考模型与官方排行榜](https://arxiv.org/abs/2306.07471) (Arxiv 2023)

## :beers: 安装

通过pip安装：

```python
pip install beir
```

如果你想从源代码构建，请使用：

```shell
git clone https://github.com/beir-cellar/beir.git
cd beir
pip install -e .
```

已在Python 3.6和3.7版本上测试通过

## :beers: 特点

- 预处理您自己的信息检索数据集或使用已经预处理好的17个基准数据集
- 包含广泛的设置，涵盖对学术界和工业界都有用的多样化基准
- 集成多种知名检索架构（词法、稠密、稀疏和重排序）
- 在一个便捷的框架中使用不同的最先进评估指标添加和评估您自己的模型

## :beers: 快速示例

有关其他示例代码，请参考我们的**[示例和教程](https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials)**Wiki页面。

```python
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#### 仅用于打印调试信息到标准输出的代码
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /打印调试信息到标准输出

#### 下载scifact.zip数据集并解压缩数据集
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### 提供scifact下载并解压缩后的data_path
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### 加载SBERT模型并使用余弦相似度进行检索
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="dot") # 或 "cos_sim" 表示余弦相似度
results = retriever.retrieve(corpus, queries)

#### 使用NDCG@k, MAP@K, Recall@K 和 Precision@K 评估您的模型，其中k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```

## :beers: 可用数据集

使用终端生成md5hash的命令：`md5sum filename.zip`。

您可以在**[这里](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/)**或**[Hugging Face](https://huggingface.co/BeIR)**查看所有可用的数据集。

| 数据集       | 网站                                                                                  | BEIR-名称          | 公开? | 类型                       | 查询 | 语料库 | 相关 D/Q |                                            下载链接                                            |                                               md5                                               |
| ------------- | ---------------------------------------------------------------------------------------- | ------------------ | ------- | -------------------------- | ------- | ------ | ------- | :---------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
| MSMARCO       | [主页](https://microsoft.github.io/msmarco/)                                         | `msmarco`          | ✅      | `train`<br>`dev`<br>`test` | 6,980   | 8.84M  | 1.1     |     [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip)      |                               `444067daf65d982533ea17ebd59501e4`                                |
| TREC-COVID    | [主页](https://ir.nist.gov/covidSubmit/index.html)                                   | `trec-covid`       | ✅      | `test`                     | 50      | 171K   | 493.5   |    [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip)    |                               `ce62140cb23feb9becf6270d0d1fe6d1`                                |
| NFCorpus      | [主页](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)                      | `nfcorpus`         | ✅      | `train`<br>`dev`<br>`test` | 323     | 3.6K   | 38.2    |     [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip)     |                               `a89dba18a62ef92f7d323ec890a0d38d`                                |
| BioASQ        | [主页](http://bioasq.org)                                                            | `bioasq`           | ❌      | `train`<br>`test`          | 500     | 14.91M | 4.7     |                                               无                                                |  [如何复现?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#2-bioasq)   |
| NQ            | [主页](https://ai.google.com/research/NaturalQuestions)                              | `nq`               | ✅      | `train`<br>`test`          | 3,452   | 2.68M  | 1.2     |        [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip)        |                               `d4d3d2e48787a744b6f6e691ff534307`                                |
| HotpotQA      | [主页](https://hotpotqa.github.io)                                                   | `hotpotqa`         | ✅      | `train`<br>`dev`<br>`test` | 7,405   | 5.23M  | 2.0     |     [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip)     |                               `f412724f78b0d91183a0e86805e16114`                                |
| FiQA-2018     | [主页](https://sites.google.com/view/fiqa/)                                          | `fiqa`             | ✅      | `train`<br>`dev`<br>`test` | 648     | 57K    | 2.6     |       [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip)       |                               `17918ed23cd04fb15047f73e6c3bd9d9`                                |
| Signal-1M(RT) | [主页](https://research.signal-ai.com/datasets/signal1m-tweetir.html)                | `signal1m`         | ❌      | `test`                     | 97      | 2.86M  | 19.6    |                                               无                                                | [如何复现?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#4-signal-1m) |
| TREC-NEWS     | [主页](https://trec.nist.gov/data/news2019.html)                                     | `trec-news`        | ❌      | `test`                     | 57      | 595K   | 19.6    |                                               无                                                | [如何复现?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#1-trec-news) |
| Robust04      | [主页](https://trec.nist.gov/data/robust/04.guidelines.html)                         | `robust04`         | ❌      | `test`                     | 249     | 528K   | 69.9    |                                               无                                                | [如何复现?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#3-robust04)  |
| ArguAna       | [主页](http://argumentation.bplaced.net/arguana/data)                                | `arguana`          | ✅      | `test`                     | 1,406   | 8.67K  | 1.0     |     [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip)      |                               `8ad3e3c2a5867cdced806d6503f29b99`                                |
| Touche-2020   | [主页](https://webis.de/events/touche-20/shared-task-1.html)                         | `webis-touche2020` | ✅      | `test`                     | 49      | 382K   | 19.0    | [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip) |                               `46f650ba5a527fc69e0a6521c5a23563`                                |
| CQADupstack   | [主页](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)                         | `cqadupstack`      | ✅      | `test`                     | 13,145  | 457K   | 1.4     |   [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip)    |                               `4e41456d7df8ee7760a7f866133bda78`                                |
| Quora         | [主页](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) | `quora`            | ✅      | `dev`<br>`test`            | 10,000  | 523K   | 1.6     |      [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip)       |                               `18fb154900ba42a600f84b839c173167`                                |
| DBPedia       | [主页](https://github.com/iai-group/DBpedia-Entity/)                                 | `dbpedia-entity`   | ✅      | `dev`<br>`test`            | 400     | 4.63M  | 38.2    |  [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip)  |                               `c2a39eb420a3164af735795df012ac2c`                                |
| SCIDOCS       | [主页](https://allenai.org/data/scidocs)                                             | `scidocs`          | ✅      | `test`                     | 1,000   | 25K    | 4.9     |     [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip)      |                               `38121350fc3a4d2f48850f6aff52e4a9`                                |
| FEVER         | [主页](http://fever.ai)                                                              | `fever`            | ✅      | `train`<br>`dev`<br>`test` | 6,666   | 5.42M  | 1.2     |      [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip)       |                               `5a818580227bfb4b35bb6fa46d9b6c03`                                |
| Climate-FEVER | [主页](http://climatefever.ai)                                                       | `climate-fever`    | ✅      | `test`                     | 1,535   | 5.42M  | 3.0     |  [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip)   |                               `8b66f0a9126c521bae2bde127b4dc99d`                                |
| SciFact       | [主页](https://github.com/allenai/scifact)                                           | `scifact`          | ✅      | `train`<br>`test`          | 300     | 5K     | 1.1     |     [链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip)      |                               `5f7d1de60b170fc8027bb7898e2efca1`                                |

## :beers: 其他信息

我们还在我们的**[Wiki](https://github.com/beir-cellar/beir/wiki)**页面提供了各种其他信息。
请参考以下页面：

### 快速开始

- [安装BEIR](https://github.com/beir-cellar/beir/wiki/Installing-beir)
- [示例和教程](https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials)

### 数据集

- [可用数据集](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [多语言数据集](https://github.com/beir-cellar/beir/wiki/Multilingual-datasets)
- [加载自定义数据集](https://github.com/beir-cellar/beir/wiki/Load-your-custom-dataset)

### 模型

- [可用模型](https://github.com/beir-cellar/beir/wiki/Models-available)
- [评估自定义模型](https://github.com/beir-cellar/beir/wiki/Evaluate-your-custom-model)

### 评估指标

- [可用评估指标](https://github.com/beir-cellar/beir/wiki/Metrics-available)

### 其他

- [BEIR排行榜](https://github.com/beir-cellar/beir/wiki/Leaderboard)
- [信息检索课程材料](https://github.com/beir-cellar/beir/wiki/Course-material-on-ir)

## :beers: 免责声明

类似于Tensorflow [datasets](https://github.com/tensorflow/datasets) 或 Hugging Face的 [datasets](https://github.com/huggingface/datasets) 库，我们只是下载并准备了公共数据集。我们仅以特定格式分发这些数据集，但我们不保证其质量或公平性，也不声称您有使用该数据集的许可。用户有责任确定您是否有权根据数据集的许可使用数据集，并引用数据集的正确所有者。

如果您是数据集所有者并希望更新其任何部分，或不希望您的数据集包含在此库中，请随时在此处发布问题或提出拉取请求！

如果您是数据集所有者并希望将您的数据集或模型包含在此库中，请随时在此处发布问题或提出拉取请求！

## :beers: 引用与作者

如果您发现此存储库有帮助，请随意引用我们的出版物 [BEIR: 用于信息检索模型零样本评估的异构基准](https://arxiv.org/abs/2104.08663)：

```
@inproceedings{
    thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}
```

如果您使用了BEIR排行榜中的任何基线分数，请随意引用我们的出版物 [酿造BEIR的资源：可复现的参考模型与官方排行榜](https://arxiv.org/abs/2306.07471)

```
@misc{kamalloo2023resources,
      title={Resources for Brewing BEIR: Reproducible Reference Models and an Official Leaderboard},
      author={Ehsan Kamalloo and Nandan Thakur and Carlos Lassance and Xueguang Ma and Jheng-Hong Yang and Jimmy Lin},
      year={2023},
      eprint={2306.07471},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

此存储库的主要贡献者是：

- [Nandan Thakur](https://github.com/Nthakur20), 个人网站: [nandan-thakur.com](https://nandan-thakur.com)

联系人: Nandan Thakur, [nandant@gmail.com](mailto:nandant@gmail.com)

如果有任何问题或发现问题，请随时给我们发送电子邮件或报告问题。

> 此存储库包含实验性软件，发布的唯一目的是提供有关相应出版物的更多背景细节。

## :beers: 合作

BEIR基准测试得以实现是由于以下大学和组织的合作努力：

- [UKP Lab, Technical University of Darmstadt](http://www.ukp.tu-darmstadt.de/)
- [University of Waterloo](https://uwaterloo.ca/)
- [Hugging Face](https://huggingface.co/)

## :beers: 贡献者

感谢所有这些精彩的合作为BEIR基准测试做出的贡献：

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://www.nandan-thakur.com"><img src="https://avatars.githubusercontent.com/u/30648040?v=4" width="100px;" alt=""/><br /><sub><b>Nandan Thakur</b></sub></a></td>
    <td align="center"><a href="https://www.nils-reimers.de/"><img src="https://avatars.githubusercontent.com/u/10706961?v=4" width="100px;" alt=""/><br /><sub><b>Nils Reimers</b></sub></a></td>
    <td align="center"><a href="https://www.informatik.tu-darmstadt.de/ukp/ukp_home/head_ukp/index.en.jsp"><img src="https://www.informatik.tu-darmstadt.de/media/ukp/pictures_1/people_1/Gurevych_Iryna_500x750_415x415.jpg" width="100px;" alt=""/><br /><sub><b>Iryna Gurevych</b></sub></a></td>
    <td align="center"><a href="https://cs.uwaterloo.ca/~jimmylin/"><img src="https://avatars.githubusercontent.com/u/313837?v=4" width="100px;" alt=""/><br /><sub><b>Jimmy Lin</b></sub></a></td>
    <td align="center"><a href="http://rueckle.net"><img src="https://i1.rgstatic.net/ii/profile.image/601126613295104-1520331161365_Q512/Andreas-Rueckle.jpg" width="100px;" alt=""/><br /><sub><b>Andreas Rücklé</b></sub></a></td>
    <td align="center"><a href="https://www.linkedin.com/in/abhesrivas"><img src="https://avatars.githubusercontent.com/u/19344566?v=4" width="100px;" alt=""/><br /><sub><b>Abhishek Srivastava</b></sub></a></td>
  </tr>
</table>


<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
