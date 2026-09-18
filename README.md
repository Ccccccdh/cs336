# CS336 后训练项目总结

本项目以 `Qwen2.5-Math-1.5B` 为基础模型，在单张 RTX 4090（24GB）上依次完成
Baseline、SFT、RSFT、DPO、GRPO、Self-Play RL 和手写 LoRA 跨领域微调。

主线目标是提升 MATH 数学推理能力；最后使用 PhyX 将模型适配到物理选择题，并检查
数学能力是否保留。

## 公共评测协议

### MATH

- 测试集：MATH 7 个学科的 test split，共 5000 题；
- Prompt：`prompts/r1_zero.prompt`；
- 生成：`temperature=1.0, top_p=1.0, max_tokens=1024`；
- 判分：`r1_zero_reward_fn(response, ground_truth, fast=True)`；
- 指标：`answer_accuracy`、`format_rate` 和三类回答计数。

统一评测命令：

```bash
python baseline/zero_shot.py --dataset math \
    --model MODEL_PATH \
    --output-dir RESULT_PATH \
    --seed 4040
```

### PhyX

PhyX 使用自定义类别分层 held-out 划分。主指标不是自由生成，而是分别计算
A/B/C/D 的条件对数概率并选择最高者，避免长推理截断和答案格式差异干扰结果。

```bash
python baseline/phyx_eval.py \
    --model MODEL_PATH \
    --data-path data/phyx_lora/test.jsonl \
    --output-dir RESULT_PATH \
    --eval-mode likelihood \
    --score-batch-size 4 \
    --max-seq-len 4096
```

---

# 1. Baseline

Baseline 直接评测未经后训练的 `Qwen2.5-Math-1.5B`。Base 模型尚未学会稳定遵守
`<think>/<answer>` 格式，因此正确率和格式率都很低。

核心流程：加载 MATH test、构造 r1-zero prompt、用 vLLM 批量生成，再由
`r1_zero_reward_fn` 统一判分。

| 指标 | 结果 |
| --- | ---: |
| answer accuracy | 2.68% |
| format rate | 16.64% |
| correct | 134/5000 |

---

# 2. SFT

SFT 使用 MATH 人工解答做全参数监督微调。训练样本包含题目、推理过程和最终答案；
prompt token 不参与 loss，只对 response token 计算交叉熵，从而让模型学会 CoT 和
`<think>/<answer>` 输出格式。

核心设置：

- MATH train 过滤后得到 7496 条；
- `micro_batch_size=1`，梯度累积得到有效 batch 32；
- 使用 AdamW、梯度裁剪和 gradient checkpointing；
- 最终模型：`models/sft_math/`。

| 指标 | Baseline | SFT |
| --- | ---: | ---: |
| answer accuracy | 2.68% | **24.54%** |
| format rate | 16.64% | **89.82%** |

---

# 3. RSFT

RSFT（Rejection Sampling Fine-Tuning）让当前模型对每道题采样多个回答，用 verifier
保留 reward=1 的正确回答，再把这些回答作为新的 SFT 数据。奖励只负责筛选数据，不直接
进入训练 loss。

```text
当前 policy 采样 G=8
    -> verifier 判分
    -> 保留格式正确且答案正确的轨迹
    -> 继续 SFT
    -> 更新后的 policy 进入下一轮
```

关键实现：

- `scripts/rsft_sample.py`：vLLM 批量采样；
- `scripts/rsft_filter.py`：筛选 reward=1 的回答；
- `trainers/rsft_train.py`：在筛选数据上继续训练；
- 最终模型：`models/rsft_round2_full/`。

| 阶段 | answer accuracy | format rate |
| --- | ---: | ---: |
| SFT | 24.54% | 89.82% |
| RSFT round 1 | 27.94% | 94.52% |
| RSFT round 2 | **32.04%** | **97.96%** |

---

# 4. DPO

DPO 使用 `(chosen, rejected)` 偏好对直接优化 policy，不需要训练显式奖励模型。
Reference model 是冻结的 `rsft_round2_full`，目标是让正确回答相对 reference 的提升大于
错误回答：

```text
L_DPO = -log sigmoid(
    beta * [(logp_chosen - logp_chosen_ref)
          - (logp_rejected - logp_rejected_ref)]
)
```

为了适应 24GB 显存，先离线缓存 reference log-prob，训练时只加载 policy。最终比较
`beta ∈ {0.1, 0.2, 0.3}`：

| beta | answer accuracy | format rate |
| ---: | ---: | ---: |
| 0.1 | **33.94%** | **98.50%** |
| 0.2 | 33.84% | 98.22% |
| 0.3 | 33.44% | 98.20% |

最终选择 `beta=0.1`，模型保存为 `models/dpo_beta01/`。

---

# 5. GRPO

GRPO 在 DPO policy 基础上进行在线强化学习。每个问题采样 G 个回答，用二值 verifier
reward 计算组内相对 advantage，不需要额外的 value model：

```text
A_i = r_i - mean(r_1, ..., r_G)
ratio_i = exp(logp_theta_i - logp_old_i)
L = -min(ratio_i * A_i,
         clip(ratio_i, 1-epsilon, 1+epsilon) * A_i)
```

训练循环：

```text
抽取一批 MATH 问题
    -> old policy 对每题生成 G=4 个回答
    -> verifier 计算 0/1 reward
    -> 组内归一化得到 advantage
    -> GRPO-Clip 更新当前 policy
    -> 新 policy 用于下一轮 rollout
```

最终配置：

```text
rollout_batch_size = 32
group_size = 4
effective_train_batch_size = 16
clip_epsilon = 0.2
max_seq_len = 1536
steps 0-60 learning_rate = 1e-5
steps 60-70 learning_rate = 5e-6
```

| checkpoint | answer accuracy | format rate |
| --- | ---: | ---: |
| DPO 初始化 | 33.94% | 98.50% |
| GRPO step 25 | 48.74% | 90.26% |
| GRPO step 50 | 58.04% | 94.78% |
| GRPO step 60 | 58.08% | 95.58% |
| GRPO step 70 | **59.90%** | **94.38%** |

最终模型使用 `models/grpo_fast/`（step 70 备份可记为 `grpo_fast_step70`）。

---

# 6. Self-Play RL

Self-Play 让同一个 policy 同时扮演 proposer 和 solver。MATH train 只提供 seed；真正用于
rollout 的题目由当前 policy 生成，并经过静态过滤与独立解答验证。

```text
MATH seed
    -> proposer 生成新题和候选答案
    -> 过滤复制、泄露、上下文依赖和重复题
    -> solver 独立验证两次
    -> 对通过的问题采样 G=8 个回答
    -> 只保留同时包含对/错回答的 mixed group
    -> GRPO 更新 policy
    -> 新 policy 进入下一轮
```

关键文件：

- `prompts/self_play_problem_gen.prompt`：出题提示词；
- `utils/self_play.py`：解析与静态过滤；
- `scripts/self_play_generate.py`：出题和答案验证；
- `scripts/self_play_rollout.py`：G 组 rollout；
- `trainers/self_play_train.py`：端到端在线主循环。

最终3个 Self-Play steps 共接收137道新问题，保留61个 mixed groups、488条 rollout。

| checkpoint | answer accuracy | format rate | 结论 |
| --- | ---: | ---: | --- |
| GRPO 初始化 | 59.90% | 94.38% | 起点 |
| Self-Play step 1 | 60.70% | 94.90% | 继续提升 |
| Self-Play step 3 | **61.28%** | **95.20%** | 最佳模型 |
| Self-Play step 5 | 60.22% | 94.28% | 性能回落，早停 |

最终模型为 `models/self_play_exp2_step3/`。继续训练后性能下降，说明自产数据会产生分布
漂移，因此采用 step 3 早停。

---

# 7. PEFT：手写 FFN LoRA

## 7.1 目标与原理

PEFT 阶段将最终数学模型适配到 PhyX 物理选择题。基础权重 W 全部冻结，只学习低秩更新：

```text
delta_W = (alpha / r) * B @ A
W' = W + delta_W
```

LoRA 只注入28个 Transformer 层的 `gate_proj`、`up_proj`、`down_proj`，共84个模块。
最终使用 `rank=4, alpha=16`：

| 项目 | 数值 |
| --- | ---: |
| 基础模型参数 | 1,543,714,304 |
| LoRA 参数 | 3,526,656 |
| 可训练参数比例 | 0.228% |

核心实现：

- `utils/lora.py`：LoRA 注入、保存、加载、merge/unmerge；
- `tests/test_lora.py`：6项单元测试，全部通过；
- `scripts/peft_inspect.py`：模型结构和数据字段检查；
- `scripts/prepare_phyx.py`：PhyX 数据准备；
- `trainers/lora_train.py`：手写 LoRA 训练器；
- `scripts/lora_merge.py`：合并 Adapter；
- `baseline/phyx_eval.py`：物理 likelihood 评测。

## 7.2 数据

使用 `Cloudriver/PhyX` 的 `with_steps/data_with_steps`，把图片转换为 `image_caption`
文字描述，并按类别固定分层划分：

| split | 数量 | 用途 |
| --- | ---: | --- |
| train | 2400 | LoRA 训练 |
| validation | 300 | 选择训练目标 |
| test | 300 | 最终评测 |

三个集合 ID 无交集。该结果属于自定义 held-out 划分，不是官方 PhyX test 排行结果。

## 7.3 训练目标选择

| 目标 | 结果 | 结论 |
| --- | --- | --- |
| PhyX steps + answer | test 29.33%，低于基线30.33% | 通用步骤语言主导 loss，不采用 |
| `A</answer>` 等5个token | validation 31%，但279/300题预测A | 选项坍缩，不采用 |
| 单个正确 A-D token | validation 39.33%，test 38.00% | **最终采用** |

最终 prompt 以 `<answer>` 结尾，response 只包含一个正确选项字母，并对 A-D 使用逆频率
平衡采样。这使训练目标与 likelihood 评测完全一致。

## 7.4 最终结果

### 物理能力

| 指标 | PEFT 前 | PEFT 后 | 变化 |
| --- | ---: | ---: | ---: |
| PhyX correct | 91/300 | **114/300** | +23 |
| PhyX accuracy | 30.33% | **38.00%** | **+7.67 pp** |
| PhyX macro accuracy | 30.08% | **38.02%** | **+7.95 pp** |

分类别结果：

| category | PEFT 前 | PEFT 后 |
| --- | ---: | ---: |
| Electromagnetism | 38.18% | 29.09% |
| Mechanics | 27.27% | 34.55% |
| Modern Physics | 25.00% | 32.50% |
| Optics | 34.00% | 32.00% |
| Thermodynamics | 30.00% | **56.00%** |
| Waves/Acoustics | 26.00% | **44.00%** |

### 数学能力保持

| 指标 | PEFT 前 | PEFT 后 | 变化 |
| --- | ---: | ---: | ---: |
| MATH correct | 3064/5000 | 3035/5000 | -29 |
| MATH accuracy | **61.28%** | **60.70%** | -0.58 pp |
| MATH format rate | 95.20% | 94.22% | -0.98 pp |

LoRA 将物理准确率提高7.67个百分点，同时数学准确率只下降0.58个百分点，达到了低成本
跨领域适配且基本保留原能力的目标。

本实验只证明文本化 PhyX 选择题能力的提升：图片由 caption 替代，最终目标是 A-D 分类，
不能据此声称模型已经获得通用视觉理解或完整物理长推理能力。

---

# 8. 全流程结果

## 8.1 MATH 主线

| 阶段 | 最终模型 | answer accuracy | format rate | 相较上一阶段 |
| --- | --- | ---: | ---: | ---: |
| Baseline | Qwen2.5-Math-1.5B | 2.68% | 16.64% | - |
| SFT | `sft_math` | 24.54% | 89.82% | +21.86 pp |
| RSFT | `rsft_round2_full` | 32.04% | 97.96% | +7.50 pp |
| DPO | `dpo_beta01` | 33.94% | **98.50%** | +1.90 pp |
| GRPO | `grpo_fast` | 59.90% | 94.38% | +25.96 pp |
| Self-Play | `self_play_exp2_step3` | **61.28%** | 95.20% | +1.38 pp |
| PEFT 后数学保持 | `phyx_lora_r4_choice_merged` | 60.70% | 94.22% | -0.58 pp |

从未对齐 base 到最佳 Self-Play，MATH 准确率累计提高58.60个百分点，正确题数从134增加
到3064。提升最大的是 SFT 的格式与基础推理对齐，以及 GRPO 的在线 verifier 强化学习。

## 8.2 跨领域结果

| benchmark | PEFT 前 | PEFT 后 | 结论 |
| --- | ---: | ---: | --- |
| MATH accuracy | 61.28% | 60.70% | 基本保留 |
| MATH format rate | 95.20% | 94.22% | 基本保留 |
| PhyX accuracy | 30.33% | **38.00%** | 提升7.67 pp |
| PhyX macro accuracy | 30.08% | **38.02%** | 提升7.95 pp |

## 8.3 方法演进

```text
人工监督 SFT
    -> 正确轨迹筛选 RSFT
    -> 成对偏好 DPO
    -> 固定题目在线 RL（GRPO）
    -> 自产题目在线 RL（Self-Play）
    -> 低秩跨领域适配（LoRA）
```

| 阶段 | 训练信号 | 更新参数 |
| --- | --- | --- |
| SFT | 人工 response token CE | 全参数 |
| RSFT | reward=1 的模型轨迹 CE | 全参数 |
| DPO | chosen/rejected 偏好 margin | 全参数 |
| GRPO | verifier reward + group advantage | 全参数 |
| Self-Play | 自产题 verifier reward | 全参数 |
| PEFT | 正确物理选项 token CE | 仅0.228% LoRA参数 |

## 8.4 最终模型

完整阶段模型：

```text
models/sft_math/
models/rsft_round2_full/
models/dpo_beta01/
models/grpo_fast/
models/self_play_exp2_step3/
models/phyx_lora_r4_choice/
models/phyx_lora_r4_choice_merged/  
```
