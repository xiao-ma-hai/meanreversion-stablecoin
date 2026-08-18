# 2026-08-18 第二、三轮反馈评估与研究结果

## 1. 总体判断

第三轮反馈保留了第二轮的全部核心 P0，并新增三稳定币共同样本、共同观测模型、同步推断与 1m--5m 半群检验。研究设计上应以第三轮为主、第二轮为其 P0 依据，而不应将两版当成两套并列任务重复执行。

两版反馈的最重要判断正确：当前 Kraken 数据有大量精确并列价格，连续 mixed-OU 点密度似然在 $\kappa\downarrow0$ 时无上界。这使旧的连续 MLE、LR、连续路径 bootstrap、Hessian 和似然切片均不能支持常规结构推断。增加 bootstrap 次数不能修复观测模型错设。

第三轮建议的三币主设计也可行。2021--2025 年 USDT、USDC、DAI 共有 229,484 个完全对齐且不插值的五分钟精确转移对，高于预设的 100,000 对门槛。不过 DAI 的自身日历覆盖率显著较低且后期下降，因此主文必须同时区分共同对齐样本和币种自身历史。

## 2. 建议可行性分类

| 建议 | 判断 | 本轮处理 |
|---|---|---|
| 并列价格导致连续 MOU 似然无上界 | 正确、P0 | 已给出理论推导、实证审计和自动退化测试；旧连续 LR 从正文撤回 |
| 潜在连续状态加 rounding/filter likelihood | 正确、P0 | 已实现有界 interval approximation 和通用网格过滤原型；生产规模稀疏/自适应过滤仍未完成 |
| 全样本结构估计与训练期预测估计分离 | 正确、P0 | 已分配置、分 CSV、分论文表格；OOS 参数固定于 2021--2023 |
| UQML 与 IFM LR 分开、quantized bootstrap | 正确，但依赖完整过滤 | 旧连续 bootstrap 作废；正式量化 LR/size/power 延后至完整过滤通过后 |
| strong persistence 必须收缩 | 正确 | 已加入 $R_{t,t}=1$ 与 $0<R_{s,t}<1$，并分离 kernel drift 与 initial-law mean convergence |
| 固定 $R$ 的 Copula 精确判据、固定时点 generator 判据 | 正确 | 已修订量词和措辞 |
| mixed local drift 需存在条件 | 正确 | 已显式加入 base drift、有限不变均值和 $a_{t,t+h}$ 的右导数条件，并用核积分证明 |
| conditional-mean acceleration 不等价于更短回归时间 | 正确 | 已改标题并加入 pathwise/variance/depeg 限制说明 |
| centered semigroup、协方差与谱隙 | 正确但需区分一般与自伴情形 | 一般情形仅给 $L^2$ 衰减界；自伴强连续情形给精确谱移位 $g_0+\lambda$ |
| CK 强制 mixture weight 乘法性 | 条件正确 | 已加入 $P_{s,t}\neq\Pi$ 的非退化条件 |
| 可测/右连续权重必为普通积分指数 | 原建议条件不足 | 已修正：右连续只给累积风险 $H$；写成 $\int\lambda$ 还需 $H$ 局部绝对连续 |
| 三币共同日历和同一 observation model | 正确且可行 | 已完成数据审计、共同样本、统一年度 tick 区间似然 |
| 三币类型回归 | 不可行/不应做 | 保留描述性异质性，不以三个币声称因果效应 |
| 同步 day-block bootstrap | 正确但计算顺序应后置 | 已实现同步抽样基础工具；参数重估待完整过滤和 size/power 通过 |
| 1m--5m 半群检验 | 正确且信息量高 | 已完成 interval-likelihood pilot；结果为明确失败，不作为稳健性成功 |
| 机制时间线 | 可行但需官方/可追溯来源 | 本轮未根据结果反向编写；列为下一阶段独立资料任务 |
| 厚尾边缘、多时域 composite likelihood | 可行且重要 | 应在完整 rounding/filter 架构上实现，优先级低于 P0 完整过滤 |

## 3. 数据审计结果

### 3.1 三币五分钟共同样本

| 资产 | 2021--2025 自身日历覆盖率 | 对齐共同转移对 | 对齐样本并列率 | 最小非零价格变化 |
|---|---:|---:|---:|---:|
| USDT | 99.8% | 229,484 | 45.3% | $10^{-5}$ |
| USDC | 92.6% | 229,484 | 54.0% | $10^{-4}$ |
| DAI | 59.3% | 229,484 | 17.1% | $10^{-5}$ |

年度审计显示 USDT 的经验 tick 在 2021--2022 年为 $10^{-4}$、2023 年后为 $10^{-5}$；USDC 为 $10^{-4}$；DAI 为 $10^{-5}$。本轮估计使用 asset-year tick，而不是为三币强制同一个数值 tick。

### 3.2 DAI 数据质量边界

DAI 五分钟覆盖率从 2021 年约 84.8% 降至 2025 年约 29.4%；一分钟覆盖率下降更明显。共同精确转移仍足以进行描述性横向比较，但 DAI 2025 OOS 推断的日聚类不确定性很大。不得 forward-fill，也不得为维持三币表格而降低数据规则。

## 4. 观测似然修复

已实现：

1. 最近 tick 量化与对数价格区间；
2. 数值稳定的 Gaussian interval probability；
3. OU 与 mixed-OU conditional interval approximation；
4. rounded Gaussian first-stage margin；
5. cell-mass grid filter 原型；
6. 连续似然退化、区间似然有界、endpoint、资产特定 tick 和量化路径测试。

interval approximation 仍把上一期潜在状态近似为观测 bin 中点，因此本轮参数只能称为“区间近似估计”。网格过滤已经可在小样本上运行，但全五年生产估计还需要 sparse/adaptive 实现。

## 5. 分样本估计和 OOS 结果

全样本表使用 2021--2025；训练表只使用 2021--2023；2024 为 validation，2025 统一称 post-selection evaluation。所有可用的精确五分钟对均用于主区间估计，不再固定系统抽取 120,000 对。

MOU 相对 OU 的冻结训练参数 OOS 平均 log-observation-probability 差异为：

| 资产 | 窗口 | 差异 | UTC 日聚类 95% CI |
|---|---|---:|---:|
| USDT | 2024 | 0.2737 | [0.2551, 0.2923] |
| USDT | 2025 post-selection | 0.3167 | [0.2982, 0.3353] |
| USDC | 2024 | 0.5577 | [0.5495, 0.5658] |
| USDC | 2025 post-selection | 0.6230 | [0.6185, 0.6275] |
| DAI | 2024 | 0.2858 | [0.0847, 0.4870] |
| DAI | 2025 post-selection | 1.6241 | [-0.2652, 3.5135] |

因此，USDT 和 USDC 的预测性混合优势清晰；DAI 2024 为正但 2025 不确定。这些结论比较的是同一观测 bin 的概率，bin width 在 MOU--OU 配对差异中完全抵消。

## 6. 三币共同样本与频率结果

共同样本 interval-MOU 点估计显示明显异质性，但 DAI 的 $\widehat\lambda$ 很大，且正式同步 bootstrap 尚未运行。因此跨币 $\Delta\kappa$、$\Delta\lambda$ 只保存点差，不报告伪造的独立标准误或显著性排序。

频率预检使用 2021--2023、按 UTC 日分层、固定种子抽取的 20,000 对样本。所有抽样索引记录 SHA-256。1 分钟动态参数转移到五分钟核后，五分钟自身估计的配对分数优势为：

| 资产 | 5m own $-$ 1m transfer | UTC 日聚类 95% CI |
|---|---:|---:|
| USDT | 0.1788 | [0.0619, 0.2957] |
| USDC | 0.1854 | [0.0722, 0.2986] |
| DAI | 0.2254 | [0.1766, 0.2743] |

三个区间均严格为正，且日尺度 $\kappa,\lambda$ 在 1m 与 5m 之间明显变化。因此半群预检失败。该失败说明年度 tick 区间近似不足以消除微观结构或单尺度模型错设，不能将五分钟参数直接解释为频率不变的连续时间结构率。

## 7. 阶段门槛状态

| 门槛 | 状态 |
|---|---|
| 含并列价格时观测 likelihood 有界 | 通过（interval approximation）；完整过滤待扩展 |
| 三币统一观测架构 | 通过 |
| 全样本与训练/OOS 完全分离 | 通过 |
| OOS 配对差异带日聚类区间 | 通过 |
| 三币共同日历且不插值 | 通过 |
| strong/local 定义、a.e. 量词、固定 $R$ | 通过 |
| 谱/协方差结论和 weight necessity | 通过严格修订 |
| production full filter | 未通过 |
| quantized LR size/power/bootstrap | 未通过；旧连续 bootstrap 已作废 |
| synchronous cross-asset refit bootstrap | 未通过 |
| 多时域 profile/Godambe identification | 未通过 |
| 1m--5m semigroup consistency | 失败（有信息的负结果） |
| 机制时间线与来源审计 | 未完成 |

## 8. 关键文件

- 数据审计：`output/tables/tick_tie_audit.csv`、`three_coin_data_audit.csv`、`common_sample_selection.csv`。
- 观测模型：`src/meanrev_stablecoin/observation/rounding.py`。
- 区间似然与过滤：`models/mou_interval_likelihood.py`、`models/mou_grid_filter.py`。
- 分样本估计：`full_sample_model_estimates.csv`、`training_model_estimates.csv`、`oos_scores_with_uncertainty.csv`。
- 三币比较：`cross_stablecoin_parameters.csv`、`cross_stablecoin_pairwise_tests.csv`。
- 频率检验：`frequency_parameter_stability.csv`、`one_to_five_minute_kernel_test.csv`。
- 修正理论：`manuscript/source/meanreversion_article_theory_revised.tex`。
- 当前论文：`submission/overleaf/main.tex`；上传包位于 `submission/copula_mean_reversion_2026-08-18_overleaf.zip`。

## 9. 下一阶段研究顺序

1. 将 grid filter 改造成 sparse/adaptive production filter，并在三个币的小样本上与当前 interval approximation 交叉验证。
2. 在过滤似然下完成 OU/MOU size 和 power 仿真，随后才运行量化 path bootstrap。
3. 联合 5/15/30/60/120/360 分钟估计，报告 profile 或 Godambe covariance，并检查 CK/multi-step calibration。
4. 在前述门槛通过后运行同步 UTC-day cross-asset refit bootstrap。
5. 再引入厚尾/半参数不变边缘；否则 margin 改进与 observation-model 修复会相互混杂。
6. 独立建立有来源约束的稳定机制时间线，不根据参数结果反向编码制度类别。
