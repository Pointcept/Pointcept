# GS2PC: 已尝试模块与效果汇总

本文档汇总目前在 Tomato GS2PC (4 类: `background/stem/leaf/flower`) 上已经尝试过的改进模块、对应配置，以及从训练日志中解析得到的效果。

---

## 1. 已实现/尝试的模块

### 1.1 Organ Expert + Gate + Fused Residual (organ-expert-fused)

目标: 把 “background 很容易” 与 “器官细分更难” 显式拆开。

- 主头: 全类分割 `main_logits (4 类)`
- 专家头: 只做 `stem/leaf/flower` 的细分类 `expert_logits (3 类)`
- 门控头: 预测当前点是否属于器官区域 `gate_logits (2 类)`
- 推理: 只对 `stem/leaf/flower` 三类 logits 做残差式修正
- 训练: `main_loss + fused_loss + expert_loss + gate_loss` (fused_loss 用来对齐训练/推理路径)

代码位置:

- `pointcept/models/default.py`: `OrganAwareResidualSegmentor`
- `pointcept/models/sparse_unet/spconv_unet_v1m1_base.py`: `SpUNet-v1m1(return_feature=True)` 返回 `seg_logits + feat`

配置:

- `configs/tomato_gs2pc/semseg-spunet-v1m1-2-tomato-gs2pc-200k-organ-expert-fused.py`

详细说明:

- `docs/gs2pc/organ_expert_fused.md`

### 1.2 Boundary-Aware 边界损失 (boundary)

目标: 提升边界区域的分类 (细结构与类间交界处)。

- 用 `kNN` 在点云上自动生成 boundary mask
- 训练时对 boundary 点额外加一项 CE

代码位置:

- `pointcept/models/default.py`: `BoundaryAwareSegmentor`

配置:

- `configs/tomato_gs2pc/semseg-spunet-v1m1-6-tomato-gs2pc-200k-boundary.py`

### 1.3 多尺度稀疏卷积块 (ms)

目标: 增强 encoder 多尺度感受野。

- 在指定 stage 插入 `MultiScaleSubMBlock` (k3 + dilated-k3 并联, concat + 1x1 fuse)

代码位置:

- `pointcept/models/sparse_unet/spconv_unet_v1m1_base.py`: `MultiScaleSubMBlock`

配置:

- `configs/tomato_gs2pc/semseg-spunet-v1m1-3-tomato-gs2pc-200k-ms.py`

### 1.4 Skip Gate (skipgate)

目标: 在 U-Net skip connection 处做选择性融合，减少低层噪声直接灌入 decoder。

代码位置:

- `pointcept/models/sparse_unet/spconv_unet_v1m1_base.py`: `skip_gate`

配置:

- `configs/tomato_gs2pc/semseg-spunet-v1m1-4-tomato-gs2pc-200k-skipgate.py`

### 1.5 Sparse SE (se)

目标: 通道重标定 (按 batch 内稀疏点的全局统计做 SE)。

代码位置:

- `pointcept/models/sparse_unet/spconv_unet_v1m1_base.py`: `SparseSEBlock`

配置:

- 在组合配置中启用: `configs/tomato_gs2pc/semseg-spunet-v1m1-5-tomato-gs2pc-200k-ms-skipgate-se.py`

### 1.6 组合: organ-expert-fused + boundary

目标: 尝试把器官专家与边界监督叠加。

- 在 `OrganAwareResidualSegmentor` 内部对 fused logits 加 boundary loss (实现为同样的 boundary mask + CE)

配置:

- `configs/tomato_gs2pc/semseg-spunet-v1m1-7-tomato-gs2pc-200k-organ-expert-fused-boundary.py`

---

## 2. 实验与指标说明

### 2.1 Ablation Tag

本次汇总对应:

- `exp/tomato_gs2pc/ablations/gs2pc_ablate_v1`
- seeds: `0, 1, 2` (共 3 次)

### 2.2 指标来源

表格中的 `final_mIoU` 与 per-class IoU 取自训练结束后的 Precise Evaluation 日志 (即 `test.py line 340/346`)，与最终推理流程一致 (包含测试阶段的分片推理与增强策略)。

生成汇总的命令:

```bash
python scripts/summarize_ablation.py --root exp/tomato_gs2pc/ablations/gs2pc_ablate_v1 --group
```

---

## 3. 结果汇总 (gs2pc_ablate_v1, 3 seeds)

说明:

- `final_mIoU_mean/std` 为 3 个 seed 的均值与标准差
- `stem/leaf/flower` 为对应类别 IoU 的均值

| variant | config | n | final mIoU (mean±std) | stem | leaf | flower |
|---|---|---:|---:|---:|---:|---:|
| baseline | `semseg-spunet-v1m1-0-...-200k.py` | 3 | 0.6215 ± 0.0015 | 0.6162 | 0.8716 | 0.0000 |
| ms | `semseg-spunet-v1m1-3-...-ms.py` | 3 | 0.6280 ± 0.0017 | 0.6362 | 0.8776 | 0.0000 |
| skipgate | `semseg-spunet-v1m1-4-...-skipgate.py` | 3 | 0.6200 ± 0.0026 | 0.6079 | 0.8740 | 0.0000 |
| ms+skipgate+se | `semseg-spunet-v1m1-5-...-ms-skipgate-se.py` | 3 | 0.6201 ± 0.0079 | 0.6090 | 0.8733 | 0.0000 |
| boundary | `semseg-spunet-v1m1-6-...-boundary.py` | 3 | 0.8159 ± 0.0149 | 0.6604 | 0.8986 | 0.7060 |
| organ-expert-fused | `semseg-spunet-v1m1-2-...-organ-expert-fused.py` | 3 | 0.8136 ± 0.0234 | 0.6840 | 0.8877 | 0.6843 |
| organ-expert-fused+boundary | `semseg-spunet-v1m1-7-...-organ-expert-fused-boundary.py` | 3 | 0.7859 ± 0.0487 | 0.6783 | 0.9000 | 0.5669 |

单次最优 run (本 tag 内):

- `organ_expert_fused_seed0`: `final_mIoU=0.8404`
- `boundary_seed0`: `final_mIoU=0.8297`

---

## 4. 关键结论 (当前观察)

- `ms/skipgate/se` 在这个 ablation 配置下没有带来提升，并且 `flower IoU=0`，说明在当前 loss/权重设置下模型对稀有类 flower 学习失败。
- `boundary` 与 `organ-expert-fused` 都显著提升了 `flower`，mIoU 进入 0.81+ 区间；两者更像是在不同形式上对 “难点区域/稀有类” 做了强化学习。
- `organ-expert-fused + boundary` 组合出现负叠加，均值下降且方差显著增大 (seed2 崩溃)，提示多损失耦合后训练稳定性变差，且对 `flower` 影响最大。

