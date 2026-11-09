# NLP 作业二：神经网络语言模型

在中文新闻语料上训练 FNN、RNN、LSTM 和 Transformer，比较字符级验证困惑度（PPL，越低越好）。

## 安装依赖

在 Python 虚拟环境中执行：

```bash
python -m pip install -r requirements.txt
```

使用 NVIDIA GPU 时，需安装支持 CUDA 的 PyTorch。可检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 准备数据

将作业语料 `news.2017.zh.shuffled.deduped.gz` 放在项目根目录，执行：

```bash
python prepare.py --data news.2017.zh.shuffled.deduped.gz
```

最后 1,000 行作为验证集，词表仅使用训练集构建。预处理结果保存到 `data/`，原始语料不包含在仓库中。

## 运行实验

训练四种模型（自动选择 CUDA、MPS 或 CPU）：

```bash
python train.py --model all --epochs 10 --seq-len 32 --embedding-dim 64 --hidden-dim 128 --layers 1 --output runs/small_10
python summarize.py --runs runs/small_10
```

参数预算接近 100 万的多种子实验（需要 CUDA）：

```bash
python matched_experiment.py
```

默认使用种子 42、43、44，每次训练 10 轮，批大小 128，上下文上限 32。结果保存在 `runs/matched_1m_3seeds/`，包含各次最佳验证 PPL、均值、样本标准差和对比图。该脚本的参数预算针对本作业语料词表设置。

已有结果时，重新汇总用 `python matched_experiment.py --summarize-only`；重新训练请通过 `--output` 指定新目录。

## 评估与检查

```bash
python evaluate.py --checkpoint runs/small_10/lstm/best.pt
python -m unittest discover -s tests -v
```

每次训练保存配置、逐轮日志和最佳权重。验证集用于选择最佳轮次，项目未设置独立测试集。数据、训练产物和报告均不提交到仓库。
