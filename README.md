# HERMIT NLU - HiERarchical MultI-Task Natural Language Understanding

*HERMIT NLU* 是一种新的神经架构，用于在口语对话系统中进行广泛的自然语言理解。 它基于分层多任务架构，提供句子含义的多层表示,包括domain、frame与frame argument（也即frame element），后两者可分别映射为意图与槽。 该架构是自注意力机制和 BiLSTM 编码器的层次结构，后跟 CRF 标记层。
<center>
	<img src="hermit_architecture.png" alt="Hermit Architecture" width="500" />
</center>

已进行了若干实验，表明这种方法在以domain和frame semantic（即frame与frame element）为注释的数据集上取得了有希望的结果。HERMIT NLU还可以模拟面向应用程序的注释方案。对公开可用的 NLU 数据集的实验表明，HERMIT 提供的总体性能高于最先进的工具，如 RASA、Dialogflow、LUIS 和 Watson。

有关详细信息，请参阅页面末尾的论文。


## 依赖及其安装

确保使用 **Python 2.7** 环境.
要运行 HERMIT NLU, 首先clone该repo:

```
git clone https://gitlab.com/hwu-ilab/hermit-nlu.git && cd hermit-nlu
```
### [可选] 虚拟环境 (建议)
建议使用conda创建虚拟环境，在成功安装conda并正确配置环境变量之后，运行:

```
conda create -n hermit-nlu python=2.7
```
然后激活该虚拟环境:

```
source activate hermit-nlu
```

### 环境安装
 `requirements.txt` 中是本实验所用到的环境，其中默认使用的是GPU版本的tensorflow，而非CPU版本，请按需更改。环境安装的命令如下:

```
pip install -r requirements.txt
```

下载并安装[spacy](https://spacy.io)的英文模型:

```
python -m spacy download en
```

### 数据预处理

实验所用的原始数据集已经clone到本repo中，下面是其原始地址：
```
https://github.com/xliuhw/NLU-Evaluation-Data
```
需要对原始数据进行预处理，命令为:

```
chmod +x data_process.sh && ./data_process.sh
```

处理之后，会生成datasets目录，且不再需要原始数据。

## 训练与测试

```
chmod +x evaluate.sh && ./evaluate.sh
``` 

注：模型评估时，会在线下载EMLo模型，而该链接可能在中国大陆无法直接访问，导致运行程序报错443，可以考虑通过https或socks5代理的方式解决，见下，填入代理的IP与端口（不加方括号）。
```
export https_proxy="socks5://[your_proxy_ip]:[your_proxy_port]"
```
或
```
export https_proxy="http://[your_proxy_ip]:[your_proxy_port]"
```

脚本将在 NLU-基准数据集上执行 10Fold 评估并生成文件夹。文件中提到文件中报告的exact match，以及 CoNLL spanF1

评测结果会保存到resource/evaluation目录中，有三个子目录，分别是encoder，用于保存每个fold的训练数据，predication，用于保存每个fold的预测结果，results，用于保存每个fold预的测指标。

总体指标由下面的命令计算：
```
chmod +x toal_metrics.sh && ./total_metrics.sh
```

## 参考文献

```

@inproceedings{vanzo:2019b,
	Address = {Stockholm, Sweden},
	Author = {Vanzo, Andrea and Bastianelli, Emanuele and Lemon, Oliver},
	Booktitle = {Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue},
	Doi = {10.18653/v1/W19-5931},
	Month = sep,
	Pages = {254--263},
	Publisher = {Association for Computational Linguistics},
	Title = {Hierarchical Multi-Task Natural Language Understanding for Cross-domain Conversational {AI}: {HERMIT} {NLU}},
	Url = {https://www.aclweb.org/anthology/W19-5931},
	Year = {2019}
	}

```
