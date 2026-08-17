from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    model: str
    family: str
    source: str
    observable: str
    density_status_target: str
    comparison_group: str
    required_by: str
    notes: str


MODEL_REGISTRY = (
    ModelSpec("OUF", "classical", "OU", "log_price", "estimable", "log_price_density", "article+methodology", "exact OU transition"),
    ModelSpec("ThresholdOU_Heteroskedastic", "classical", "threshold OU", "log_price", "estimable", "log_price_density", "methodology", "state-asymmetric local Gaussian transition"),
    ModelSpec("CIR_Price", "classical", "CIR", "price", "estimable", "log_price_density", "article+methodology", "price density converted to log-price density"),
    ModelSpec("CIR_DepegPressure", "classical", "CIR", "absolute_depeg", "estimable", "amplitude_only", "methodology", "not invertible to signed price"),
    ModelSpec("AffineTransformedOU", "OU_transform", "OU Example 1", "log_price", "estimable", "log_price_density", "article+methodology", "affine monotone transform"),
    ModelSpec("ExponentialTransformedOU", "OU_transform", "OU Example 2", "price", "attempt", "log_price_density", "article+methodology", "positive exponential-affine transform"),
    ModelSpec("SpecialFunctionTransformedOU", "OU_transform", "OU Example 3", "price", "diagnostic", "structural_failure", "article", "even in latent OU state; generally not globally monotone"),
    ModelSpec("AffineTransformedCIR", "CIR_transform", "CIR Example 1", "price", "estimable", "log_price_density", "article+methodology", "affine monotone transform"),
    ModelSpec("QuadraticTransformedCIR", "CIR_transform", "CIR Example 2", "price", "attempt", "log_price_density", "article+methodology", "monotone only when A(t)>0 and B(t)>=0"),
    ModelSpec("ExponentialTransformedCIR", "CIR_transform", "CIR Example 3", "price", "attempt", "log_price_density", "article+methodology", "positive exponential-affine transform"),
    ModelSpec("SpecialFunctionTransformedCIR", "CIR_transform", "CIR Example 4", "price", "attempt", "log_price_density", "article", "numerical inverse and monotonicity audit required"),
    ModelSpec("MOUF", "mixed_kernel", "Poisson mixed OU", "log_price", "estimable", "log_price_density", "article+methodology", "joint constant-intensity fit"),
    ModelSpec("MCIR", "mixed_kernel", "Poisson mixed CIR", "price", "estimable", "log_price_density", "article+methodology", "base CIR weak-identification propagates"),
    ModelSpec("SeasonalIntensityMOU", "mixed_kernel", "deterministic c(t)", "log_price", "attempt", "log_price_density", "article+methodology", "positive seasonal deterministic reset intensity"),
    ModelSpec("JumpOU", "benchmark", "jump OU", "log_price", "attempt", "log_price_density", "methodology", "compound-jump transition approximation explicitly labelled"),
    ModelSpec("MarkovSwitchingOU", "benchmark", "two-regime OU", "log_price", "attempt", "log_price_density", "methodology", "hidden-state likelihood"),
    ModelSpec("NestedOU", "benchmark", "two latent OU factors", "log_price", "attempt", "log_price_density", "methodology", "exact linear-Gaussian state-space likelihood"),
    ModelSpec("RandomWalk", "benchmark", "no reversion", "log_price", "estimable", "log_price_density", "methodology", "strong no-reversion benchmark"),
    ModelSpec("GaussianGammaStationary", "copula_margin", "Gaussian copula + stationary Gamma", "positive_state", "attempt", "copula_density", "article", "article gamma Case 1 sign condition audited"),
    ModelSpec("GaussianGammaTimeVarying", "copula_margin", "Gaussian copula + time-varying Gamma", "positive_state", "attempt", "copula_density", "article", "article gamma Case 2 ODE restriction audited"),
    ModelSpec("ParametricCopulaGrid", "copula_margin", "parametric margins and copulas", "log_price", "attempt", "log_price_density", "methodology", "all declared margin x copula x reset combinations"),
    ModelSpec("TwoScaleGaussian", "latent_factor", "two latent OU factors", "log_price", "estimable", "copula_density", "empirical_extension", "scalar observation is not one-dimensional Markov"),
    ModelSpec("TwoScaleMixed", "latent_factor", "two latent OU factors + reset", "log_price", "estimable", "copula_density", "empirical_extension", "latent-vector Markov extension"),
    ModelSpec("FullyTimeVaryingMarginalCopula", "time_varying", "time-varying marginal-generator model", "log_price", "attempt", "log_price_density", "methodology", "finite-dimensional deterministic trend/seasonal specification"),
    ModelSpec("MixedCopulaBondPDE", "pricing", "mixed-copula bond PDE", "bond_prices", "not_observable", "not_comparable", "article", "no bond cross section or risk-neutral measure in stablecoin data"),
)


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(spec) for spec in MODEL_REGISTRY])

