# NLP 作业二：四种神经网络语言模型

在同一份中文新闻语料上，从头训练 FNN、普通 RNN、LSTM 和 Transformer，比较字符级困惑度（PPL）。

## 1. 环境

在此文件夹打开终端。如果当前 Python 已安装 PyTorch 和 matplotlib，可以直接进入下一步：

```bash
python3 -c "import torch, matplotlib; print(torch.__version__)"
```

如果缺少依赖，建议建立独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

建议 Python 3.9 或更新版本。`--device auto` 按 CUDA、MPS、CPU 的顺序选择设备。如果 Mac 没有检测到 MPS，会自动用 CPU。可以通过 `--threads 4` 调整 CPU 线程数。

## 2. 处理数据（只需一次）

```bash
python3 prepare.py
```

默认读取当前文件夹中的 `news.2017.zh.shuffled.deduped`，保留原始顺序，最后 1,000 行作为验证集。也可以直接读取压缩文件：

```bash
python3 prepare.py --data news.2017.zh.shuffled.deduped.gz
```

本地数据统计如下：

| 项目 | 数值 |
|---|---:|
| 训练句子 | 163,125 |
| 验证句子 | 1,000 |
| 训练字符 | 6,574,053 |
| 验证字符 | 40,367 |
| 词表大小（含 4 个特殊符号） | 4,840 |
| 验证集预测位置（字符 + EOS） | 41,367 |

词表只用训练集建立，默认将训练集中频率小于 2 的字符映射为 `<unk>`。这样未知字符也有训练样本。保留句子内的空格和标点，不进行分词或繁简转换。

输出：`data/processed.pt`、`data/processed.stats.json`、`data/processed.vocab.json`。

## 3. 先检查流程

协议检查覆盖数据划分、未来信息屏蔽、片段目标覆盖和 PPL 计算：

```bash
python3 -m unittest discover -s tests -v
```

用少量批次检查四个模型能否完成训练和保存：

```bash
python3 train.py --model all --output runs/smoke --epochs 1 --batch-size 4 --seq-len 16 --embedding-dim 16 --hidden-dim 32 --layers 1 --heads 2 --limit-train-sentences 32 --max-train-batches 2 --max-val-batches 2 --log-every 1
python3 summarize.py --runs runs/smoke
```

**这些是流程检查结果，不能作为作业正式实验结果。** 如输出文件夹已有结果，选择新的 `--output`；确定要替换时加 `--overwrite`。

## 4. 正式训练

一次训练四个模型，每个模型 5 轮：

```bash
python3 train.py --model all --epochs 5 --output runs/main
```

也可以分开执行，方便在每个模型完成后查看结果：

```bash
python3 train.py --model fnn --epochs 5 --output runs/main
python3 train.py --model rnn --epochs 5 --output runs/main
python3 train.py --model lstm --epochs 5 --output runs/main
python3 train.py --model transformer --epochs 5 --output runs/main
```

上面两种方式任选一种。比较同一组实验时，保持训练参数一致。

默认配置：

| 参数 | 默认值 |
|---|---:|
| 上下文长度上限 | 64 |
| 批大小 | 32 |
| 字符嵌入维度 | 128 |
| FNN/RNN/LSTM 隐藏维度、Transformer 前馈维度 | 256 |
| RNN/LSTM 层数、Transformer 层数 | 2 |
| 注意力头数 | 4 |
| Dropout | 0.1 |
| Adam 学习率 | 0.001 |
| 梯度裁剪阈值 | 1.0 |
| 随机种子 | 42 |

每轮会显示训练损失、验证 PPL 和耗时，每 100 个训练批次输出进度。每个模型单独保存验证损失最低的权重。

如果 CPU 训练较慢，可先在新的文件夹做较小的**全量数据实验**：

```bash
python3 train.py --model all --epochs 5 --seq-len 32 --embedding-dim 64 --hidden-dim 128 --layers 1 --output runs/small
```

这种设置仍然使用完整训练集和最后 1,000 句验证集，可以用于报告，但要写明实际配置。先观察首轮耗时再估计总时长。需要改变配置时请使用新输出文件夹。

## 5. 汇总与独立评估

```bash
python3 summarize.py --runs runs/main
python3 evaluate.py --checkpoint runs/main/lstm/best.pt
```

小模型实验则将 `runs/main` 换成 `runs/small`。

汇总生成：

- `comparison.csv`：参数量、最佳验证 PPL、最佳轮次、训练耗时。
- `comparison.md`：可直接参考的结果表。
- `learning_curves.png`：训练损失与验证 PPL 曲线。

每个模型文件夹包括 `config.json`、`history.csv`、`best.pt`、`summary.json`。有限批次或有限训练句子的实验会明确标为 DEBUG。未完成所有轮次的汇总也会提示。

## 6. 实验协议与报告说明

- 每句编码为 BOS、原始字符、EOS；按最多 64 个预测位置切片。一个目标字符只计算一次，不拼接不同句子。超长句在片段边界重置上下文，片段开头的可用历史较短，四个模型遵守相同规则。
- 所有模型在位置 t 预测下一个字符。FNN 使用最多 64 个历史字符嵌入的拼接，经线性层、tanh、Dropout 和输出层预测；代码用等价的因果卷积实现，避免展开所有重叠窗口占用过多内存。
- 普通 RNN 使用 tanh 循环单元；LSTM 使用门控单元。二者维度和层数相同，在每个片段重置隐藏状态。
- Transformer 使用可学习位置编码、因果自注意力和前馈层。Transformer 的模型维度为字符嵌入维度，前馈维度为 `hidden_dim`。
- PPL = exp(所有有效目标的交叉熵之和 / 有效目标数)。PAD 和 BOS 不计入目标，句末 EOS 计入。验证集完整评估应有 41,367 个目标；不会直接平均批次 PPL。
- 四种模型参数量不完全相等，代码记录实际值。这是一组共享数据与上下文协议的架构对比，报告中需同时给出参数量。固定随机种子不保证不同硬件间逐位一致。
- 每轮使用验证集选取最优模型；本作业没有独立测试集，因此报告使用“验证集 PPL”，不称为测试集成绩。
- 主实验先保持同一组超参数。如果以后针对各模型调整学习率，单独记录调参过程，避免混淆实验设置。

## 文件结构

```text
prepare.py           数据处理入口
train.py             统一训练入口
evaluate.py          最佳模型独立评估
summarize.py         结果表和曲线
nlp_lm/data.py       字符词表、编码、片段和补齐
nlp_lm/models.py     FNN、RNN、LSTM、Transformer
nlp_lm/engine.py     训练、验证和困惑度
tests/test_protocol.py 关键实验协议检查
```
