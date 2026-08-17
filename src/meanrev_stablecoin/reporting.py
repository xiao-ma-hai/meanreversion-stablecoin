from __future__ import annotations

import hashlib
import html
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels

from .constants import project_root
from .data_io import resolve_path, write_csv, write_json


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _text_table(df: pd.DataFrame, columns: list[str] | None = None, rows: int = 20) -> str:
    if df.empty:
        return "（尚无可用结果）"
    view = df if columns is None else df[[column for column in columns if column in df.columns]]
    return "```text\n" + view.head(rows).to_string(index=False) + "\n```"


def compile_model_tables(tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimate_files = sorted(tables.glob("model_estimates_*_baseline.csv"))
    comparison_files = sorted(tables.glob("model_comparison_*_baseline.csv"))
    estimates = pd.concat([pd.read_csv(path) for path in estimate_files], ignore_index=True) if estimate_files else pd.DataFrame()
    comparisons = pd.concat([pd.read_csv(path) for path in comparison_files], ignore_index=True) if comparison_files else pd.DataFrame()
    if not estimates.empty:
        estimates = estimates.drop_duplicates(["sample", "model", "parameter"], keep="last")
        write_csv(estimates, tables / "model_estimates.csv")
    if not comparisons.empty:
        comparisons = comparisons.drop_duplicates(["sample", "model"], keep="last")
        calculated_acceptance = (
            comparisons["converged"].fillna(False).astype(bool)
            & comparisons["gradient_norm"].lt(1e-4)
            & ~comparisons["weak_identification"].fillna(False).astype(bool)
        )
        if "acceptance_pass" not in comparisons:
            comparisons["acceptance_pass"] = calculated_acceptance
        else:
            comparisons["acceptance_pass"] = comparisons["acceptance_pass"].where(
                comparisons["acceptance_pass"].notna(), calculated_acceptance
            )
        write_csv(comparisons, tables / "model_comparison.csv")
    return estimates, comparisons


def build_empirical_report(config: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    root = project_root()
    tables = resolve_path(config["output"]["tables"])
    figures = resolve_path(config["output"]["figures"])
    models = resolve_path(config["output"]["models"])
    logs = resolve_path(config["output"]["logs"])
    estimates, comparisons = compile_model_tables(tables)
    audit = _read(tables / "data_audit.csv")
    sample_summary = _read(tables / "sample_summary.csv")
    sign = _read(tables / "sign_test_baseline.csv")
    copula = _read(tables / "copula_horizon_fit.csv")
    forecast = _read(tables / "forecast_scores.csv")
    validation = _read(tables / "validation_model_selection.csv")
    events = _read(tables / "depeg_events.csv")
    robustness = _read(tables / "robustness_estimates.csv")
    block_robustness = _read(tables / "ou_block_length_robustness.csv")
    lr = _read(tables / "mixed_lr_bootstrap.csv")
    matched_lr = _read(tables / "matched_joint_mou_lr_bootstrap.csv")
    scale_models = _read(tables / "copula_scale_model_comparison.csv")
    scale_oos = _read(tables / "copula_scale_oos_scores.csv")
    model_hierarchy = _read(tables / "article_model_hierarchy.csv")
    first_passage = _read(tables / "first_passage_comparison.csv")
    full_status = _read(tables / "full_model_implementation_status.csv")
    full_ranking = _read(tables / "full_model_validation_ranking.csv")
    full_pairwise = _read(tables / "full_model_pairwise_validation.csv")
    full_gates = _read(tables / "updated_stage_gate.csv")

    cir_numerical_rows = []
    cir_path = models / "cir_fits_baseline.json"
    if cir_path.exists():
        for fit in json.loads(cir_path.read_text(encoding="utf-8")):
            cir_numerical_rows.append({
                "model": fit["model_name"],
                "density_backend": fit["metadata"].get("transition_density_backend"),
                "scipy_exact_failure_rate": fit["metadata"].get("scipy_exact_logpdf_failure_rate"),
                "gradient_norm": fit.get("gradient_norm"),
                "weak_identification": fit["metadata"].get("weak_identification"),
            })
    cir_numerical = pd.DataFrame(cir_numerical_rows)
    if not cir_numerical.empty:
        write_csv(cir_numerical, tables / "cir_numerical_diagnostics.csv")

    important_params = estimates[estimates["parameter"].isin(["theta", "kappa", "sigma", "lambda"])] if not estimates.empty else estimates
    comparison_cols = ["model", "nobs", "loglik", "aic", "bic", "converged", "weak_identification", "acceptance_pass", "pit_ks_pvalue", "ljung_box_z_p20"]
    forecast_test = forecast[forecast["period"] == "test_2025"] if not forecast.empty else forecast
    baseline_events = events[events.get("specification", pd.Series(dtype=str)).eq("baseline_50_25bp")] if not events.empty else events

    report = f"""# USDT/USD 五分钟数据均值回复实证报告

生成日期：2026-08-09  
数据：Kraken `USDTUSD_5.csv`，UTC，2017-03-29 至 2025-12-31。  
主样本：2021-2025；训练/验证/测试：2021-2023 / 2024 / 2025。

## 1. 执行结论

本项目按两份 TeX 方法论建立了只读原始数据审计、精确时间匹配、非参数漂移、OU/阈值 OU、CIR、全部有限维理论变换、MOU/MCIR、全部声明的边际--copula 组合、强基准、边界 bootstrap、统一验证排名、滚动预测、脱锚事件和首次回复时间流程。只有 `pytest`、原 MVP 与更新后全模型阶段门槛均为 PASS 时，manifest 才会标记项目完成。

环境采用 `D:/Anaconda/envs/meanreversion-stablecoin`。由于本机 classic 与 libmamba 求解器均长期停留在索引阶段，最终环境通过对现有 Python 3.11 科学栈进行只读离线克隆到新名称，再仅向新环境补装 PyYAML、pytest、arch 与本项目包；没有修改或删除任何原有环境。离线克隆带入的少数非项目包存在既有 `pip check` 冲突，已原样记录在环境审计中；它们不属于本项目依赖且未被实证代码导入。`environment.yml` 是干净重建规范，两个导出文件记录实际运行环境。

## 2. 数据审计与样本

{_text_table(audit, ["metric", "observed", "expected", "match"], 30)}

{_text_table(sample_summary, ["sample", "start_utc", "end_utc", "observations", "theoretical_bars", "coverage_pct", "exact_5min_pairs", "hard_anomalies"], 10)}

三条 2019 年候选异常始终保留原值并显式标记；全样本稳健性同时报告保留和排除标记记录。主模型只把时间差严格等于 300 秒的观测组成五分钟转移对，不把相邻 CSV 行无条件当作相邻 K 线。

## 3. 非参数均值回复证据

{_text_table(sign, ["theta_definition", "theta", "SCR", "SCR_bootstrap_ci_lower", "SCR_bootstrap_ci_upper", "Tn_equal_weight", "Tn_density_weight"], 10)}

局部线性漂移使用按 UTC 日分块的交叉验证与 1,000 次最终 bootstrap；逐点区间和同时置信带均已保存。SCR 是有限网格上的方向一致率，不能被解释成对连续时间极限或全状态空间均值回复的证明。

## 4. 参数模型与诊断

{_text_table(important_params, ["model", "parameter", "estimate", "ordinary_se", "sandwich_se", "bootstrap_se", "bootstrap_ci_lower", "bootstrap_ci_upper", "weak_identification", "half_life_minutes", "effective_half_life_minutes"], 80)}

{_text_table(comparisons, comparison_cols, 30)}

普通似然模型的 AIC/BIC 可在相同观测与似然口径下比较；半参数复合似然不与普通似然直接按原始 AIC/BIC 排名。CIR 和仿射变换若 Hessian/轮廓显示弱识别，报告保留估计但不作精确结构解释。

CIR 数值密度诊断：

{_text_table(cir_numerical, None, 10)}

价格 CIR 在极高非中心参数区间出现 SciPy 非中心卡方对数密度数值失效，故以 CIR 条件一、二阶矩的 Gaussian 极限作数值回退，并把高回退率强制标成弱识别；这不是“精确 CIR 似然”的证据。脱锚压力 CIR 未触发该回退。

## 5. 混合 copula 与边界检验

{_text_table(lr, None, 10)}

## 5A. Article-core extensions (2026-08-07)

Primary matched-design joint-MOUF boundary bootstrap:
{_text_table(matched_lr, ["matched_pairs", "replications", "observed_lr", "finite_replication_corrected_pvalue", "bootstrap_alternative_convergence_rate", "bootstrap_lambda_boundary_share"], 5)}

Prespecified model hierarchy:
{_text_table(model_hierarchy, None, 10)}

Single- versus two-scale copula comparison:
{_text_table(scale_models, ["model", "n_parameters", "objective_mean_log_copula", "dependence_rmse", "kappa", "kappa_slow", "kappa_fast", "weight_fast", "lambda"], 10)}

Frozen-training-model equal-horizon out-of-sample scores:
{_text_table(scale_oos[scale_oos["horizon_minutes"].astype(str).eq("equal_horizon_mean")] if not scale_oos.empty else scale_oos, ["period", "model", "n_pairs", "mean_log_copula_score"], 20)}

The two-scale latent state is Markov, but its observed scalar sum is generally
not one-dimensional Markov. These models are dependence diagnostics, not new
instances of the article's scalar Markov proposition.

{_text_table(copula, ["horizon_minutes", "n_pairs", "empirical_gaussian_rho", "fitted_base_rho", "fitted_mixed_linear_dependence", "reset_weight"], 20)}

MOU 的 `kappa` 是基础依赖衰减，`lambda` 是 Poisson 型独立重置强度；有效条件均值半衰期由 `kappa+lambda` 决定。二者只靠条件均值不可分别识别，因此使用完整混合密度、多滞后 copula 和边界 bootstrap。`lambda` 不自动对应某个唯一经济事件。

## 6. 样本外预测、脱锚概率与回复时间

验证期选择：

{_text_table(validation, None, 10)}

2025 测试期评分：

{_text_table(forecast_test, ["model", "horizon_minutes", "threshold", "nobs", "MAE", "RMSE", "NLS", "CRPS", "Brier", "event_log_score", "AUC"], 80)}

基准 50/25bp 事件数：{len(baseline_events)}。

{_text_table(first_passage, None, 10)}

随机游走是无回复强基准；`NestedOU_proxy` 使用慢变指数均值加快 OU 偏离，是辅助文档要求的多层 OU 思路的可复现代理，但不是完整状态空间 Kalman 多层 OU，因而不应把它的表现当作对所有嵌套 OU 模型的最终否定。

## 7. 稳健性

{_text_table(robustness, ["sample", "anomaly_policy", "frequency_minutes", "n_pairs", "theta", "kappa", "lambda", "half_life_minutes", "effective_half_life_minutes"], 30)}

UTC 日移动块长度稳健性：

{_text_table(block_robustness, None, 20)}

稳健性覆盖 2020-2025 扩展样本、15/30/60 分钟完整聚合区间的精确滞后、1/3/7 日移动块，以及 2017-2025 全样本的异常保留/排除口径。频率变化不仅改变抽样误差，也可能降低微观结构噪声；若 `kappa` 随降频明显下降，应优先解释为高频噪声偏误风险。

## 8. 对数学模型与实证方法的独立评估

### 合理且可执行的部分

1. 以 `log(P)` 表示相对一美元锚的有符号偏离是自然且可解释的；OU 有精确不规则间隔转移密度，是最强的首要基准。
2. 直接检验 `(theta-x)mu(x)>0` 比只报告 ADF/AR(1) 更贴近文档的均值回复定义；时间哈希配对避免了缺失 K 线造成的伪五分钟转移。
3. 若基础核 `P_h` 以稳态边缘 `F` 为不变分布，则 `Q_h=exp(-lambda h)P_h+[1-exp(-lambda h)]Pi_F` 对应 Poisson 重置过程，并保持 Markov 半群，数学构造成立。Gaussian copula 的 `rho_h=exp(-kappa h)` 加经验单调边缘也可由潜在 Gaussian OU 变换实现。
4. 训练、验证、测试按时间切分，测试伪观测用训练期经验分布映射，配合密度评分和脱锚概率，能防止最明显的未来信息泄漏。

### 必须谨慎解释或需要后续改进的部分

1. 有限的五分钟漂移符号和样本均值稳定性只是连续时间定义的代理，不能证明 `t→∞` 的矩收敛，也不能排除结构突变、状态切换或非 Markov 记忆。
2. 成交收盘价的最小报价单位、买卖价差和 bid-ask bounce 相对 USDT 的真实波动很大，可能显著高估 `kappa`。当前数据没有中间价/订单簿，降频稳健性只能缓解、不能识别有效价格与测量误差；正式论文应加入噪声状态空间 OU。
3. CIR 直接拟合约 1 的价格时 `sqrt(P)` 几乎常数，和 OU 很难区分；CIR 拟合绝对偏离又丢失方向，且绝对值过程未必本身满足 CIR。它适合作为比较/风险幅度模型，不宜成为核心价格机制。
4. 仿射变换的 `A0`—潜在波动缩放、`d`—`kappa` 方差趋势和 `B0`—长期均值均可能近共线；多年样本的指数尺度还依赖时间原点。固定 `A0=1`、先固定 `B0=0` 与报告 profile likelihood 是必要但未必充分的识别约束。
5. MOU 的重置从稳态边缘独立抽样，能描述依赖中断或数据跳变，但真实脱锚往往是状态依赖、方向不对称且持续的冲击。常数 `lambda` 可能把数据错误、微观噪声、套利断裂和新闻跳跃混为一体；下一版宜估计状态依赖 `lambda(x,z_t)`、非对称重置边缘或跳跃/状态切换基准。
6. `kappa` 与 `lambda` 的条件均值只识别其和；小间隔下重置概率很低，边界附近的有限样本分布非标准。参数 bootstrap、多滞后信息和本项目的 1/3/7 日块结果是必要诊断，但 capped bootstrap 和复合似然仍应在论文定稿前用更大算力及模拟识别实验复核。
7. 365 日滚动窗口和每 30 日重估是合理的预注册折衷，但会平滑短危机并制造高度重叠估计；结构变化结论必须与成交量、交易笔数和外部事件共同验证，不能只看滚动参数跳动。
8. 单一 Kraken USDT/USD 无法支持跨稳定币、跨交易所或因果机制外推。它足以完成模型可估计性与预测比较，但不能把 `lambda` 解释为赎回、储备或链上行为的因果效应。

### 总体可行性判断

MVP 在现有 OHLCVT 数据上可行，最可信的结果层级依次为：数据事实与精确配对 → 非参数方向检验与 OU/阈值 OU → 样本外概率预测 → MOU/半参数 copula 的统计增益 → CIR/仿射变换的结构解释。前四层可形成论文主实证；最后一层只有在识别诊断、噪声模型和外部数据补齐后才适合作强经济解释。

## 8A. Full-model registry and validation selection (2026-08-09)

All finite-dimensional article models and all methodology benchmarks are
estimated or retain a terminal failure diagnostic before validation ranking.

{_text_table(full_gates, ["gate", "pass", "criterion"], 30)}

Top unified 2024 five-minute conditional-density scores:

{_text_table(full_ranking, ["validation_log_score_rank", "model", "mean_log_score", "finite_score_coverage"], 15)}

Paired day-block uncertainty for the selected model:

{_text_table(full_pairwise, None, 10)}

Retained weak/failure diagnoses:

{_text_table(full_status[full_status["weak_identification"].fillna(False).astype(bool) | ~full_status["implementation_status"].astype(str).str.startswith("estimated")] if not full_status.empty else full_status, ["model", "implementation_status", "weak_identification", "failure_reason"], 30)}

## 9. 复现与产物

最终 CSV 位于 `output/tables`，PNG/PDF 图位于 `output/figures`，模型 JSON/NPZ 位于 `output/models`，UTF-8 日志位于 `output/logs`。运行状态见 `output/run_status.json` 与 `output/manifest.json`。
"""
    md_path = root / "output/empirical_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report, encoding="utf-8")
    html_path = root / "output/empirical_report.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>USDT/USD mean reversion report</title>"
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1100px;margin:2rem auto;line-height:1.55}pre{overflow:auto;background:#f5f5f5;padding:1rem}</style>"
        f"<pre>{html.escape(report)}</pre>", encoding="utf-8"
    )

    pip_check = subprocess.run(
        [sys.executable, "-B", "-m", "pip", "check"],
        text=True, capture_output=True, check=False,
    )
    environment_audit = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "dedicated_environment_path_expected": r"D:\Anaconda\envs\meanreversion-stablecoin\python.exe",
        "dedicated_environment_verified": str(Path(sys.executable)).lower() == r"d:\anaconda\envs\meanreversion-stablecoin\python.exe".lower(),
        "environment_creation_note": "Offline read-only clone into a new target; missing packages installed only in target.",
        "project_required_imports_verified_by_pipeline": True,
        "pip_check_returncode": pip_check.returncode,
        "pip_check_output": (pip_check.stdout + pip_check.stderr).strip(),
        "pip_check_note": "Reported conflicts are inherited packages outside pyproject project dependencies; preserved rather than mutating unrelated packages.",
    }
    write_json(environment_audit, logs / "environment_audit.json")
    shutil.copy2(root / "configs/base.yml", models / "base_config_snapshot.yml")

    artifacts = []
    for path in sorted((root / "output").rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": config["data"]["expected_sha256"],
        "random_seed": config["project"]["random_seed"],
        "artifacts": artifacts,
        "tables_count": len(list(tables.glob("*.csv"))),
        "png_count": len(list(figures.glob("*.png"))),
        "pdf_count": len(list(figures.glob("*.pdf"))),
    }
    write_json(manifest, root / "output/manifest.json")
    return md_path, html_path, manifest
