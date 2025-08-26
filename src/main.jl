"""
    StratifiedHoldout(; fraction_train=0.7, shuffle=nothing, rng=nothing)

Stratified holdout resampling strategy that maintains target distribution
in both training and test sets.

# Parameters
- `fraction_train::Float64=0.7`: Fraction of data for training (0 < fraction_train < 1)
- `shuffle::Bool=false`: Whether to shuffle before stratification
- `rng::Union{Int,AbstractRNG}=Random.default_rng()`: Random number generator or seed

# Example
```julia
X, y = @load_iris
holdout = StratifiedHoldout(fraction_train=0.8, shuffle=true, rng=123)
evaluate(DecisionTreeClassifier(), X, y, resampling=holdout, measure=accuracy)
```
"""
struct StratifiedHoldout <: MLJBase.ResamplingStrategy
    fraction_train::Float64
    shuffle::Bool
    n_bins::Int
    rng::AbstractRNG

    function StratifiedHoldout(
        fraction_train::Float64, shuffle::Bool, n_bins::Int, rng::AbstractRNG
    )
        0 < fraction_train < 1 || error("`fraction_train` must be between 0 and 1.")
        return new(fraction_train, shuffle, n_bins, rng)
    end
end

# Keyword constructor with smart defaults
function StratifiedHoldout(;
    fraction_train::Float64=0.7,
    shuffle::Bool=false,
    n_bins::Int=5,
    rng=Random.default_rng(),
)
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    return StratifiedHoldout(fraction_train, shuffle, n_bins, rng)
end

# Main implementation - requires target variable y for stratification
function MLJBase.train_test_pairs(strategy::StratifiedHoldout, rows, y)
    # Determine task type from the scitype of the full target vector
    is_finite = scitype(y) <: AbstractVector{<:Union{Missing,Finite}}

    if is_finite
        return stratified_holdout_classification(strategy, rows, y)
    else
        return stratified_holdout_regression(strategy, rows, y)
    end
end

# Classification: Maintain class proportions
function stratified_holdout_classification(strategy::StratifiedHoldout, rows, y)
    # Optional shuffle
    rows_working =
        strategy.shuffle ? rows[randperm(strategy.rng, length(rows))] : collect(rows)

    y_subset = y[rows_working]
    class_counts = countmap(y_subset)

    # Handle edge case of single class
    if length(class_counts) == 1
        train, test = partition(
            rows_working, strategy.fraction_train; shuffle=false, rng=strategy.rng
        )
        return [(train, test)]
    end

    train_indices = Int[]
    test_indices = Int[]

    # Stratify each class
    for (class, count) in class_counts
        class_indices = findall(x -> isequal(x, class), y_subset)
        n_train = max(1, round(Int, count * strategy.fraction_train))
        n_train = min(n_train, count - 1)  # Ensure at least 1 test sample

        if count < 2
            append!(train_indices, class_indices)
        else
            train_class = sample(strategy.rng, class_indices, n_train; replace=false)
            test_class = setdiff(class_indices, train_class)
            append!(train_indices, train_class)
            append!(test_indices, test_class)
        end
    end

    # Convert back to original indices
    train_global = rows_working[train_indices]
    test_global = rows_working[test_indices]

    # Fallback if one of the splits is empty after stratification
    if isempty(train_global) || isempty(test_global)
        train_global, test_global = MLJBase.partition(
            rows_working, strategy.fraction_train; shuffle=false, rng=strategy.rng
        )
    end

    return [(train_global, test_global),]
end

# Regression: Use quantile-based stratification
function stratified_holdout_regression(strategy::StratifiedHoldout, rows, y)
    rows_working =
        strategy.shuffle ? rows[randperm(strategy.rng, length(rows))] : collect(rows)

    y_subset = y[rows_working]

    # Create quantile bins on non-missing values
    nonmissing_mask = .!ismissing.(y_subset)
    y_nonmissing = collect(skipmissing(y_subset))
    n_samples = length(y_nonmissing)
    n_bins = min(strategy.n_bins, n_samples ÷ 2)
    quantiles = range(0, 1; length=n_bins+1)
    boundaries = [quantile(y_nonmissing, q) for q in quantiles]

    # Handle duplicate boundaries
    boundaries = unique(boundaries)
    n_bins = length(boundaries) - 1

    if n_bins < 2
        # Fall back to random split if insufficient variation
        train, test = MLJBase.partition(
            rows_working, strategy.fraction_train; shuffle=false, rng=strategy.rng
        )
        return [(train, test),]
    end

    train_indices = Int[]
    test_indices = Int[]

    # Stratify within each bin
    for i in 1:n_bins
        if i == n_bins
            lower_ok = nonmissing_mask .& (y_subset .>= boundaries[i])
            upper_ok = nonmissing_mask .& (y_subset .<= boundaries[i + 1])
        else
            lower_ok = nonmissing_mask .& (y_subset .>= boundaries[i])
            upper_ok = nonmissing_mask .& (y_subset .< boundaries[i + 1])
        end

        bin_mask = lower_ok .& upper_ok
        bin_indices = findall(bin_mask)
        n_bin = length(bin_indices)

        if n_bin <= 1
            append!(train_indices, bin_indices)
        else
            n_train = max(1, round(Int, n_bin * strategy.fraction_train))
            n_train = min(n_train, n_bin - 1)

            train_bin = sample(strategy.rng, bin_indices, n_train; replace=false)
            test_bin = setdiff(bin_indices, train_bin)

            append!(train_indices, train_bin)
            append!(test_indices, test_bin)
        end
    end

    # Handle missing values as their own stratum
    missing_indices = findall(.!nonmissing_mask)
    if !isempty(missing_indices)
        if length(missing_indices) == 1
            append!(train_indices, missing_indices)
        else
            n_train_miss = max(
                1,
                min(
                    length(missing_indices) - 1,
                    round(Int, length(missing_indices) * strategy.fraction_train),
                ),
            )
            train_miss = sample(strategy.rng, missing_indices, n_train_miss; replace=false)
            test_miss = setdiff(missing_indices, train_miss)
            append!(train_indices, train_miss)
            append!(test_indices, test_miss)
        end
    end

    train_global = rows_working[train_indices]
    test_global = rows_working[test_indices]

    # Fallback if one of the splits is empty after stratification
    if isempty(train_global) || isempty(test_global)
        train_global, test_global = MLJBase.partition(
            rows_working, strategy.fraction_train; shuffle=false, rng=strategy.rng
        )
    end

    return [(train_global, test_global),]
end

# Fallback for compatibility (warns and uses random split)
function MLJBase.train_test_pairs(strategy::StratifiedHoldout, rows)
    @warn "StratifiedHoldout requires target variable y for stratification. " *
        "Using random holdout instead."
    train, test = MLJBase.partition(
        rows, strategy.fraction_train; shuffle=strategy.shuffle, rng=strategy.rng
    )
    return [(train, test),]
end
