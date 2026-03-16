# Organ-Expert-Fused: 面向 Stem/Leaf/Flower 的专家残差修正

本文档说明 `organ-expert-fused` 这一版模型相对 SpUNet baseline 的具体改动点、训练/推理逻辑、配置方式与注意事项。

适用任务: Tomato GS2PC 语义分割 (4 类: `background/stem/leaf/flower`)。

---

## 1. 动机: 显式拆开“容易的 background”和“更难的器官细分”

在 4 类分割里，`background` 往往是占比大且容易的类；而 `stem/leaf/flower` 三个器官类更相似、更细碎、更难分。

`organ-expert-fused` 的核心思想是:

1. **主头 (main head)** 仍然做全类别分割，保持原始 SpUNet 的建模能力与稳定性。
2. 增加一个只关注器官子集的 **细分类专家头 (expert head)**，把难任务集中到更小的 label space。
3. 增加一个 **门控头 (gate head)** 学习 “当前点是否属于器官区域”，避免专家输出在背景区域产生副作用。
4. 推理时不替换全量 logits，而是对 `stem/leaf/flower` 的 logits 做**残差式修正**，并且仅在 gate 判为器官时生效。

---

## 2. 模型结构改动

### 2.1 新增的 Segmentor: `OrganAwareResidualSegmentor`

实现位置:

- `pointcept/models/default.py` 里的 `OrganAwareResidualSegmentor`

它包裹一个 backbone，并额外带两个轻量线性头:

- `expert_head: Linear(backbone_out_channels -> K)`，其中 `K = len(organ_class_ids)`，在 GS2PC 里 K=3 (stem/leaf/flower)
- `gate_head: Linear(backbone_out_channels -> 2)`，二分类 (non-organ vs organ)

对 backbone 的额外要求:

- backbone 必须返回 `dict(seg_logits=..., feat=...)`，其中 `feat` 是每个点的最终特征向量 (用于 expert/gate)。

为此，SpUNet 增加了 `return_feature` 开关:

- `pointcept/models/sparse_unet/spconv_unet_v1m1_base.py`:
  - `SpUNet-v1m1(..., return_feature=True)` 时返回 `{"seg_logits": seg_logits, "feat": feat}`

### 2.2 输出与融合 (Residual Fuse)

设:

- `main_logits ∈ R^{N×C}`: 全类别 logits (C=4)
- `expert_logits ∈ R^{N×K}`: 器官子集 logits (K=3)
- `gate_logits ∈ R^{N×2}`: 是否为器官的二分类 logits
- `organ_class_ids = [1, 2, 3]` 对应 `stem/leaf/flower`
- `gate_prob = softmax(gate_logits)[:, organ] ∈ R^{N×1}`

推理时对器官类做残差修正:

1. 取出主头在器官三类上的 logits:

`organ_main = main_logits[:, organ_class_ids] ∈ R^{N×K}`

2. 计算残差 (默认使用 `residual_mode="interpolate"`):

`expert_residual = expert_logits - organ_main`

3. 只更新器官三类 logits:

`fused_logits = main_logits.clone()`

`fused_logits[:, organ_class_ids] = organ_main + gate_prob * expert_residual * residual_scale`

直观理解:

- 当 `gate_prob≈0` (更像 background) 时，`fused≈main`，专家几乎不影响结果。
- 当 `gate_prob≈1` (更像器官) 时，`fused≈expert` (仅在器官子空间内)，相当于让专家接管器官细分类。
- `residual_scale` 用于控制专家修正强度 (默认 1.0)。

实现细节:

- 为避免 AMP 下 `Half/Float` dtype 冲突，融合时会把更新项 cast 到 `fused_logits.dtype` 再写回。

---

## 3. 训练目标 (Training Objectives)

`organ-expert-fused` 是**多任务训练**，训练时对三个头分别监督，并额外对最终 fused 输出做监督 (避免训练/推理不一致)。

### 3.1 主任务损失 (Main)

- 标准语义分割 CE:
  - `main_loss = CE(main_logits, segment)`

### 3.2 专家损失 (Expert)

把全局 label 映射为器官局部 label:

- 对于 `segment == stem/leaf/flower`:
  - stem -> 0
  - leaf -> 1
  - flower -> 2
- 其它点 -> `ignore_index` (不参与 expert 监督)

然后做 K 类 CE:

- `expert_loss = CE(expert_logits, expert_target)`

### 3.3 门控损失 (Gate)

构造二分类标签:

- 非 ignore 的点默认标为 `0` (non-organ)
- 若 `segment ∈ organ_class_ids` 标为 `1` (organ)
- 若 segment 在 `gate_ignore_class_ids` 或本身为 ignore，则 gate 也 ignore

然后做二分类 CE:

- `gate_loss = CE(gate_logits, gate_target)`

### 3.4 融合输出损失 (Fused)

训练时额外对 `fused_logits` 做一次和主任务一致的 CE:

- `fused_loss = CE(fused_logits, segment)`

这一步是 `organ-expert-fused` 的关键点之一:

- 推理用的是 fused logits，不是 main logits。
- 如果只监督 main/expert/gate，而不监督 fused，可能出现 “训练时各头各自最优，但融合后输出不可控” 的 mismatch。

### 3.5 总损失

`loss = w_main*main_loss + w_fused*fused_loss + w_expert*expert_loss + w_gate*gate_loss`

在默认配置中:

- `w_main = 1.0`
- `w_fused = 1.0`
- `w_expert = 0.5`
- `w_gate = 0.2`

---

## 4. 配置与使用方式

对应配置文件:

- `configs/tomato_gs2pc/semseg-spunet-v1m1-2-tomato-gs2pc-200k-organ-expert-fused.py`

关键配置项:

- `model.type = "OrganAwareResidualSegmentor"`
- `organ_class_ids=[1, 2, 3]`
- `backbone.type="SpUNet-v1m1"`
- `backbone.return_feature=True` (否则拿不到 `feat`，expert/gate 无法工作)
- `backbone_out_channels=96` (与 SpUNet 最终 `feat` 维度一致)

训练命令示例:

```bash
conda activate pointcept
cd /mnt/data/yyd/Pointcept

python tools/train.py \
  --config-file configs/tomato_gs2pc/semseg-spunet-v1m1-2-tomato-gs2pc-200k-organ-expert-fused.py \
  --num-gpus 1 \
  --options seed=0 save_path=exp/tomato_gs2pc/spunet_gs2pc_200k_organ_expert_fused_seed0
```

测试命令示例 (用最优 checkpoint):

```bash
python tools/test.py \
  --config-file configs/tomato_gs2pc/semseg-spunet-v1m1-2-tomato-gs2pc-200k-organ-expert-fused.py \
  --num-gpus 1 \
  --options weight=exp/tomato_gs2pc/spunet_gs2pc_200k_organ_expert_fused_seed0/model/model_best.pth
```

---

## 5. 这套改动“本质上改了什么”

相对 baseline (SpUNet + 单一 CE) 的变化可以总结为:

1. **显式任务分解**
   - 主任务: background vs organs + organs 粗细分都在一个 head 里完成。
   - 现在把 “器官细分” 单拎出来作为专家子任务。

2. **受控的专家介入**
   - 专家不是硬替换输出，只在器官三类 logits 上做 residual 修正。
   - gate 让专家“只在需要的时候出手”，避免对背景的负迁移。

3. **推理路径与训练目标对齐**
   - fused 输出在训练时有直接监督 (`fused_loss`)，让最终用于评测/推理的路径可被显式优化。

4. **极小的额外计算**
   - 只新增两个 `Linear` 头和少量融合计算，对吞吐影响很小。

---

## 6. 常见坑与排查

1. 训练时出现 “requires backbone to return dict with seg_logits and feat”
   - 说明 backbone 没开 `return_feature=True`，或用了不支持该返回的 backbone。

2. AMP 下 dtype 报错 (Half/Float 不匹配)
   - 融合写回时需保证 dtype 一致。本实现已处理，若你自行改融合公式请保留 cast。

3. gate 全程趋近 0 或 1
   - 可能 `gate_loss_weight` 太大/太小，或 gate 的 class weight 不合适。
   - 可检查训练日志里的 `gate_loss` 是否收敛、是否过大。

4. 专家 head 过拟合/不学习
   - 专家目标只覆盖器官点，如果器官点很少，建议适当提高 `expert_loss_weight` 或使用更合理的 organ 类权重。

---

## 7. 进一步可做的消融 (建议)

如果你要写论文或进一步稳健验证，推荐至少做:

- 去掉 `fused_loss` (只保留 main/expert/gate) 看推理 mismatch 影响多大
- 只加 expert 不加 gate (gate_prob 固定为 1) 看 gate 的必要性
- 只加 gate 不加 expert (用 gate 做 background vs organ 二阶段) 作为对照
- 不同 `residual_scale` 或 residual_mode 对结果的影响

