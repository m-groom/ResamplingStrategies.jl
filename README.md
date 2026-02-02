# ResamplingStrategies.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Julia](https://img.shields.io/badge/julia-1.9%20%7C%201.10%20%7C%201.11-9558B2.svg)](https://julialang.org)

**ResamplingStrategies.jl** provides additional resampling strategies for [MLJ](https://github.com/JuliaAI/MLJ.jl). The current focus is on stratified resampling methods for both classification and regression tasks.

## Overview

When evaluating machine learning models, it's crucial that train/test splits preserve the statistical properties of the target variable. `ResamplingStrategies` implements stratified resampling that ensures representative sampling across all target values, reducing variance in performance estimates and preventing biased evaluation on imbalanced datasets.

### Key Features

- **Classification stratification**: Maintains exact class proportions in train/test splits
- **Regression stratification**: Uses quantile-based binning to preserve target distribution
- **Robust edge case handling**: Handles missing values, single-class datasets, and small sample sizes
- **MLJ integration**: Implements the standard `ResamplingStrategy` interface for seamless integration

### StratifiedHoldout

The `StratifiedHoldout` strategy extends MLJ's standard `Holdout` by ensuring that:
- **Classification**: Each class appears in train/test splits proportionally to its frequency
- **Regression**: Target values are binned into quantiles, and stratification is applied within bins
- **Missing values**: Treated as a separate stratum in regression tasks

This is particularly valuable for imbalanced datasets where random splits might produce unrepresentative train/test sets.

## Installation

```julia
using Pkg
Pkg.add("ResamplingStrategies")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/ResamplingStrategies.jl")
```

## Quick Start

```julia
using MLJBase
using MLJDecisionTreeInterface
using StatisticalMeasures
using ResamplingStrategies

# Load imbalanced classification data
X, y = @load_iris

stratified_holdout = StratifiedHoldout(fraction_train=0.8, rng=123)

# Evaluate model with stratified resampling
model = DecisionTreeClassifier(max_depth=3)
mach = machine(model, X, y)

# Stratified evaluation ensures all classes are represented
result = evaluate!(
    mach,
    resampling=stratified_holdout,
    measure=LogLoss()
)

println("Performance: ", result.measurement)
println("Per-fold results: ", result.per_fold)

# Also works with regression via quantile binning
X_reg, y_reg = @load_boston
model_reg = DecisionTreeRegressor()
mach_reg = machine(model_reg, X_reg, y_reg)

evaluate!(
    mach_reg,
    resampling=StratifiedHoldout(fraction_train=0.8, n_bins=5),
    measure=RootMeanSquaredError()
)
```

## Licence

This software is distributed under the MIT Licence.
