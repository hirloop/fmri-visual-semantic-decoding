# 从 fMRI 脑活动解码视觉语义文本

> 复刻学习记录：Doerig et al., *"High-level visual representations in the human
> brain are aligned with large language models"*, **Nature Machine Intelligence
> (2025)** —— 用 LLM 句子嵌入作为中间表示，从被试观看自然图片时的 fMRI 信号
> 解码出图片描述文本。

## 一句话原理

被试看图片时，视觉皮层的活动模式包含了图片的语义内容。这个语义内容可以用
"图片描述文本在 LLM 嵌入空间中的向量"来定量刻画。于是解码分两步：

```
fMRI 体素活动 ──岭回归──▶ 768 维 MPNet 句子嵌入 ──字典最近邻──▶ 图片描述文本
```

训练数据来自 [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org)：
被试 1 观看了 10,000 张 COCO 自然图片（每张 3 次，共 30,000 个 trial），
每张图有 5 条人工描述。**fMRI→嵌入**用岭回归学习，**嵌入→文本**不做生成，
而是在大规模描述语料里做最近邻查找——这正是原论文的方案。

## 项目管线与进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0. 文本侧先行 | MPNet 嵌入 + 字典查找微缩演示（噪声鲁棒性扫描） | ✅ 完成 |
| 1. 数据准备 | NSD subj01 元数据 + betas 下载（并行断点续传） | ✅ 元数据全量，betas 进行中 |
| 2. 目标矩阵 | 10,000 图 × 5 描述 → Y 矩阵（10,000×768） | ✅ 完成，0 缺失 |
| 3. 解码训练 | streams ROI 体素 → 嵌入的岭回归（CV 选 λ） | ✅ 已跑通（当前 9/40 session） |
| 4. 交互演示 | 选图 → 载入 fMRI → 预测嵌入 → 查字典 → Top-5 文本 | ✅ 完成 |
| 5. 完整复刻 | 40 session 全量训练 + CC12M 大字典 + shared-515 测试集 | ⏳ 数据下载中 |

## 阶段结果

### 文本侧：嵌入 + 字典查找的噪声鲁棒性（`src/step1_embedding_lookup.py`）

模拟"解码嵌入"= 目标嵌入 + 高斯噪声，在 30,012 条描述的字典里查最近邻：

| 目标余弦 | hit@1 | hit@5 | 说明 |
|---|---|---|---|
| 1.00 | 0.849 | 0.968 | 无噪声上限（受同义句干扰限制） |
| 0.80 | 0.852 | 0.960 | 几乎不衰减 |
| 0.60 | 0.824 | 0.959 | 重噪声下仍稳定命中 |

**关键发现**：MPNet 空间中"同场景描述"聚得很紧，与干扰句之间间隔很大——
这解释了为什么论文用简单字典查找（而非生成式 LLM）就足以重建描述。
定性例子：真值 *"a man walking and holding a surfboard on his head"* 在
余弦 0.6 噪声下检索回 *"A man who is carrying a surfboard over his head."*

### fMRI 解码（`src/decode_nsd.py` / `src/train_model.py`）

| 训练量 | 测试 cosine（均值） | 备注 |
|---|---|---|
| 1 session / 363 trial（冒烟） | ≈ 0.14 | 机制验证，无个体辨别力（预期内） |
| **9 session / ~5,900 trial（当前模型）** | **0.183**（中位 0.178） | λ=1.0（3 折 CV 选出），10,000 体素 |

个体差异很大：解码-真值余弦最高到 **0.68**，真值图在 10,000 张候选图中的排名
最好到 **#9**（随机期望 ~#5000）。定性例子（真实解码输出，非挑选理论值）：

| 真值描述（被试看的图） | 字典检索 Top-1 | 余弦 / 真值排名 |
|---|---|---|
| A black cat sits in a car and looks out. | there is a cat that is looking out of the window | +0.578 / **#9** |
| A husky dog has an orange frisbee in it's mouth. | A dog stands in the grass next to a stuffed bear. | +0.447 / #70 |

完整量级（论文为约 27,000 训练 trial + 分数岭回归）待 40 session 下载完成后训练。

## 交互解码程序（`src/decode_app.py`）

![交互解码程序截图](docs/app_screenshot.png)

*截图为真实运行状态：选中图片（解码-真值余弦 0.677 的样例）点击开始解码后，
左侧显示**被试观看的原图片**、真值人工描述与 fMRI 信号溯源，右侧为字典检索
Top-5 文本及余弦分数。*

```bash
python src/fetch_test_images.py   # 先抽取测试图片（zip 按需抽取，无需整包下载）
python src/decode_app.py          # 需先完成数据准备与模型训练（见下）
```

操作流程：

1. 下拉框选择**岭回归模型**（`outputs/ridge_models/` 下的模型包）
2. 下拉框选择**被试看过的图片**（显示 cocoId 与第一条人工描述）
3. 点击 **开始解码**：程序自动载入该图片各次重复的 fMRI beta
   （并显示来源 session/trial 溯源），经岭回归预测嵌入向量，
   在字典中做最近邻查找
4. 界面展示：真值描述（左）、**匹配度最高的 5 条文本及余弦分数**（右）、
   解码诊断（预测-真值余弦、逐次重复一致性、真值图在 10,000 张候选图中的排名）

无 GUI 环境（或想快速自检）可运行 `python src/decode_app.py --selftest`。

## 目录结构

```
├── README.md                  # 本文件
├── docs/DATA_STRUCTURE.md     # ★ 数据与目录结构详解（尤其 NSD fMRI 数据）
├── src/
│   ├── step1_embedding_lookup.py     # 文本侧演示：嵌入 + 字典查找 + 噪声扫描
│   ├── list_nsd_s3.py                # NSD S3 桶清单查询工具
│   ├── download_nsd.py               # NSD 并行断点续传下载器（16 连接）
│   ├── precompute_caption_embeddings.py  # Y 矩阵预计算
│   ├── decode_nsd.py                 # 解码管线（加载→对齐→岭回归→评估）
│   ├── train_model.py                # 训练并保存模型包（供交互程序使用）
│   ├── fetch_test_images.py          # 按需抽取测试图片（zip HTTP-Range 抽取）
│   ├── decode_app.py                 # ★ 交互解码程序（Tkinter，含图片展示）
│   └── watch_and_decode.py           # 看门狗：数据齐后自动全量解码
├── results/                    # 小体积结果文件（大文件见 .gitignore）
└── .gitignore
```

数据（fMRI 25GB+ / 字典 3GB / 模型 438MB）不入库，
完整结构与获取方式见 [docs/DATA_STRUCTURE.md](docs/DATA_STRUCTURE.md)。

## 复现指南

```bash
# 环境：Python 3.10+，CPU 即可（MPNet 嵌入 4 线程约 30 条/秒）
pip install numpy requests nibabel sentence-transformers

# 1. 下载 NSD subj01 最小数据集（免 AWS 账号，S3 公开桶，约 21.6GB）
python src/download_nsd.py

# 2. 下载 COCO 描述标注（captions_train2017.json / captions_val2017.json）
#    与 MPNet 模型（sentence-transformers/all-mpnet-base-v2）并放入 data/、models/

# 3. 预计算解码目标 Y 矩阵
python src/precompute_caption_embeddings.py

# 4. 训练解码模型（用已下载的 session）
python src/train_model.py

# 5. 抽取测试图片（从 COCO zip 镜像按需 Range 抽取约 800 张，无需下整包）
python src/fetch_test_images.py

# 6. 交互解码
python src/decode_app.py
```

## 踩坑记录（真实顺序）

1. **73KID 是 1-indexed**：responses.tsv 的 `73KID = nsdId + 1`。
   不验证直接当 0-indexed 用，对齐会错一位、只匹配上一小部分图。
2. **beta 值 -32768 是缺失标记**，不是极端激活，读取时必须置零/掩膜。
3. **NSD 同时提供 hdf5（43.8GB）与 nii.gz（19.6GB）两种格式**：
   nii.gz 有 gzip 压缩反而更小；hdf5 利于随机访问但没必要。
   下载器最初没过滤，差点下双份。
4. **高斯噪声标定**：给嵌入加噪声时，σ 是"每维度"标准差——768 维叠加后
   噪声能量放大 768 倍。第一版噪声参数直接把信噪比打到纯噪声水平，
   检索全灭。正确做法是按目标余弦反推 σ（`noise_std_for_cos()`）。
5. **S3 单流仅 ~80KB/s**：靠 HTTP Range 16 连接并行拉到 ~1MB/s+；
   下载器必须有分块级断点续传（中途断网/限流时已下分块不丢）。
6. **同图三次重复会泄漏**：训练/测试必须按"图"划分，不能按 trial 划分。
7. **COCO train/val 都要**：NSD 有约 4% 的图来自 val2017。

## 与论文的差异（当前简化）

| 项目 | 论文 | 本复刻（当前） |
|---|---|---|
| 正则化 | 分数岭回归，20 档分数，逐特征，5 折 CV | 单一全局 λ，3 折 CV |
| 体素 | streams ROI 全部 27,638 | R2 前 10,000 |
| 测试集 | 515 张全体被试共享图 | 随机 800-1,000 张图（按图划分） |
| 字典 | Google Conceptual Captions 310 万条 | COCO 描述 30,012 条（缓存版） |
| 训练量 | ~27,000 trial | ~6,000 trial（数据仍在下载） |

## 后续计划

- [ ] 40 个 session 下载完成后全量重训（watcher 已挂，自动触发基线解码）
- [ ] 换用 CC12M 大字典（本地已有 1,027 万条文本，嵌入需 ~20h CPU 或批处理）
- [ ] 分数岭回归（逐特征正则化）对齐论文
- [ ] shared-515 测试集 + noise ceiling（5 描述内部一致性）
- [ ] 与论文官方代码（[visuo_llm](https://github.com/adriendoerig/visuo_llm)）对数

## 致谢

- 原论文：Doerig, Kietzmann, Allen, Wu, Naselaris, Kay & Charest (2025)
- 数据：[Natural Scenes Dataset](https://naturalscenesdataset.org)（CC BY-NC-SA）
  与 [COCO Captions](https://cocodataset.org)
- 嵌入模型：[sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
