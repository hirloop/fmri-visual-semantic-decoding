# 数据与目录结构说明

本项目的完整工作目录在本地为 `visuo_llm_repro/`（25GB+，不入库），
GitHub 仓库只包含代码、文档和小体积结果。数据可以按 README 的说明重新下载。

## 一、工作目录总览

```
visuo_llm_repro/
├── visuo_llm/                  # 论文官方代码（GitHub 克隆，MIT，仅参考）
├── data/
│   ├── coco/
│   │   ├── captions_train2017.json   # COCO 训练集描述标注（91MB）
│   │   │                              # 118,287 图 × ~5 条人工描述，含 image_id
│   │   └── captions_val2017.json     # COCO 验证集描述（4.8MB，5,000 图）
│   │                                  # NSD 的图有 4% 来自 val2017，必须要有
│   ├── gcc/
│   │   ├── cc12m_captions.tsv        # CC12M 字典语料（2.96GB，1,027 万条文本，
│   │   │                              # 字典查找的候选句池，可扩容）
│   │   └── coco_captions.jsonl.gz    # COCO 描述备用纯文本版（6MB）
│   └── nsd/                          # ★ NSD fMRI 数据（见第二节详解）
├── models/
│   └── all-mpnet-base-v2/            # MPNet 句子嵌入模型（438MB，768 维输出）
├── src/                              # 全部自写代码（见 README 脚本表）
├── outputs/                          # 运行产出（见第三节）
└── README.md                         # 本地进度记录
```

## 二、NSD fMRI 数据结构详解

下载自 AWS S3 公开桶 `s3://natural-scenes-dataset`（免账号），
只下载了 subj01 的最小必需子集（约 21.6GB，全量数据集是 TB 级 8 被试）。

```
data/nsd/
├── nsddata/                                    # 元数据层（"账本"）
│   ├── experiments/nsd/
│   │   ├── nsd_stim_info_merged.csv            # ★ 73,000 张图的总索引
│   │   │     列含义:
│   │   │       nsdId      0~72999，NSD 内部图片编号
│   │   │       cocoId     对应的 COCO 图片 ID（查描述标注的外键）
│   │   │       cocoSplit  train2017 / val2017
│   │   │       subject1   subj01 是否看过此图（=1）
│   │   │       subject1_rep0/1/2  三次重复各自的 trial 全局序号
│   │   └── nsd_expdesign.mat                   # 实验设计矩阵
│   └── ppdata/subj01/
│       ├── behav/
│       │   └── responses.tsv                   # ★ 行为账本：30,000 行 = 每个 trial
│       │         列含义:
│       │           SESSION/RUN/TRIAL  trial 的时空定位
│       │           73KID     该 trial 看的图片编号（= nsdId + 1，1-indexed！）
│       │           ISCORRECT/RT       识别任务正确性与反应时
│       └── func1pt8mm/                         # 1.8mm 体积空间的所有辅助图
│           ├── roi/                            # ★ ROI 掩膜（39 个文件）
│           │     streams.nii.gz                #   解码用：视觉通路 ROI，
│           │     │                             #   值 1-7 = 7 个视觉区，0 = 区域外
│           │     │                             #   共 27,638 个有效体素
│           │     nsdgeneral/prf-*/floc-*/...   #   其他可选 ROI 定义
│           ├── R2.nii.gz                       # ★ 每体素信噪比（选可靠体素用）
│           ├── meanbeta*.nii.gz                # 平均 beta 图（未用于解码）
│           └── ncsnr*.nii.gz                   # 噪声 CEILING SNR 图
│
└── nsddata_betas/                              # fMRI 信号层（"数据本体"）
    └── ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/
        ├── betas_session01..40.nii.gz          # ★★ 核心：40 个 session 的 beta 值
        │     文件格式: NIfTI (gzip)，shape = (81, 104, 83, 750)
        │       前三维 = 1.8mm 全脑体积（699,192 体素）
        │       第四维 = 750 个 trial 的单试次 beta 权重
        │       dtype = int16；**-32768 = 该体素无数据**（读取时置零）
        │     语义: GLM 拟合 + GLMdenoise + RR 正则后的单试次激活强度
        │           （这是论文指定使用的预处理版本）
        └── FRACvalue_session*.nii.gz           # 分数岭回归的正则化参数（小文件）
```

### 对齐链（本项目最关键的一张图）

```
betas_sessionXX.nii.gz 第 t 个体积          （脑活动，什么时候）
        │  responses.tsv 按行序对齐: session XX 的第 t 个 trial
        ▼
73KID  ──(减 1)──▶  nsdId                    （看的是哪张图）
        │  nsd_stim_info_merged.csv 查表
        ▼
cocoId + cocoSplit                            （图在 COCO 里的身份）
        │  captions_{train,val}2017.json 查表
        ▼
5 条人工描述 ──MPNet 嵌入取平均──▶ 768 维向量   （图的"语义真值" Y）
```

两个易踩的坑：
1. **73KID 是 1-indexed**（= nsdId + 1），已用 10,000/10,000 全匹配验证；
2. beta 文件里的 **-32768 不是数值**，是缺失标记。

## 三、outputs/ 产出文件

| 文件 | 大小 | 内容 |
|---|---|---|
| `nsd_subj01_caption_embeddings.npz` | 28MB | ★ Y 矩阵：10,000 图 × 768 的目标嵌入（含 nsdId/cocoId 索引） |
| `embed_cache_5000_1000.npz` | 97MB | ★ 字典缓存：30,012 条描述的嵌入 D + 原文（交互程序用它查字典） |
| `ridge_models/ridge_sXX_vYYYY_日期.npz` | ~50MB | ★ 训练好的岭回归模型包（权重 + 测试集 beta + 真值描述 + 溯源信息） |
| `step1_results.npz` | 1KB | 第 1 步噪声扫描数值结果 |
| `decode_pred_test.npz` | ~5MB | 解码管线的预测嵌入与评估数据 |
| `nsd_download.log` / `*_run.log` | 小 | 下载与各次运行的日志 |

## 四、内存/性能备忘

- MPNet 嵌入：CPU 4 线程约 30 条/秒（50,000 条 ≈ 28 分钟）
- betas 加载：单个 session 解压约 30-40 秒（gzip 整体解压，无法随机访问）
- 岭回归（10,000 体素）：Gram 矩阵 400MB float32，CPU 可承受
- 交互程序已在保存模型时**预提取**测试 trial 的 beta，因此查图解码是秒级的
