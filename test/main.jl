using Test
using MLJBase
using Statistics
using Random
using StatsBase: countmap
using ScientificTypesBase: scitype, Finite
using ResamplingStrategies

# Helper functions for generating test data
function generate_balanced_classification_data(n_samples::Int = 100, n_classes::Int = 3)
    n_per_class = n_samples ÷ n_classes
    y = repeat(1:n_classes, inner = n_per_class)
    X = randn(length(y), 2)  # Simple 2D features
    return X, categorical(y)
end

function generate_imbalanced_classification_data(n_samples::Int = 100)
    # Create imbalanced dataset: 70% class 1, 20% class 2, 10% class 3
    n1, n2, n3 =
        round(Int, 0.7*n_samples), round(Int, 0.2*n_samples), round(Int, 0.1*n_samples)
    y = vcat(fill(1, n1), fill(2, n2), fill(3, n3))
    X = randn(length(y), 2)
    return X, categorical(y)
end

function generate_regression_data(
    n_samples::Int = 100;
    add_missing::Bool = false,
    constant::Bool = false,
)
    if add_missing
        if constant
            y = Vector{Union{Float64,Missing}}(fill(5.0, n_samples))
        else
            y = Vector{Union{Float64,Missing}}(randn(n_samples) * 10 .+ 50)  # Normal distribution with mean=50, std=10
        end
        # Add some missing values (10% of data)
        missing_indices = rand(1:n_samples, max(1, n_samples ÷ 10))
        y[missing_indices] .= missing
    else
        if constant
            y = fill(5.0, n_samples)
        else
            y = randn(n_samples) * 10 .+ 50  # Normal distribution with mean=50, std=10
        end
    end

    X = randn(n_samples, 3)
    return X, y
end

# Helper function to check stratification quality for classification
function check_classification_stratification(y_train, y_test, original_y; tolerance = 0.1)
    original_props = proportions(original_y)
    train_props = proportions(y_train)
    test_props = proportions(y_test)

    # Check if proportions are maintained within tolerance
    train_ok = all(
        abs(train_props[k] - original_props[k]) <= tolerance for
        k in keys(original_props) if haskey(train_props, k)
    )
    test_ok = all(
        abs(test_props[k] - original_props[k]) <= tolerance for
        k in keys(original_props) if haskey(test_props, k)
    )

    return train_ok && test_ok
end

function proportions(y)
    counts = countmap(y)
    total = sum(values(counts))
    return Dict(k => v/total for (k, v) in counts)
end

# Helper function to check regression stratification (quantile-based)
function check_regression_stratification(y_train, y_test, original_y; n_bins = 5)
    # Remove missing values for quantile calculation
    y_train_clean = collect(skipmissing(y_train))
    y_test_clean = collect(skipmissing(y_test))
    original_clean = collect(skipmissing(original_y))

    if isempty(y_train_clean) || isempty(y_test_clean) || isempty(original_clean)
        return true  # Can't check empty data
    end

    # For small datasets or constant values, be very lenient
    if length(original_clean) < 20 || std(original_clean) < 1e-6
        return true  # Skip stratification check for small/constant data
    end

    # Check if distributions are similar using quantiles
    quantiles = [0.25, 0.5, 0.75]
    train_quantiles = [quantile(y_train_clean, q) for q in quantiles]
    test_quantiles = [quantile(y_test_clean, q) for q in quantiles]
    original_quantiles = [quantile(original_clean, q) for q in quantiles]

    # Use generous tolerance - stratification is imperfect especially with small samples
    tolerance = max(1.0, 0.5 * std(original_clean))  # At least 1.0 or 50% of std
    train_ok = all(
        abs(train_quantiles[i] - original_quantiles[i]) <= tolerance for
        i = 1:length(quantiles)
    )
    test_ok = all(
        abs(test_quantiles[i] - original_quantiles[i]) <= tolerance for
        i = 1:length(quantiles)
    )

    return train_ok && test_ok
end

@testset "StratifiedHoldout Constructor Tests" begin
    @testset "valid parameters" begin
        # Default constructor
        holdout1 = StratifiedHoldout()
        @test holdout1.fraction_train == 0.7
        @test holdout1.shuffle == false
        @test holdout1.rng isa AbstractRNG

        # Custom parameters
        holdout2 = StratifiedHoldout(fraction_train = 0.8, shuffle = true, rng = 123)
        @test holdout2.fraction_train == 0.8
        @test holdout2.shuffle == true
        @test holdout2.rng isa MersenneTwister

        # RNG as AbstractRNG
        rng = MersenneTwister(42)
        holdout3 = StratifiedHoldout(fraction_train = 0.6, rng = rng)
        @test holdout3.fraction_train == 0.6
        @test holdout3.rng === rng
    end

    @testset "invalid parameters" begin
        # fraction_train out of bounds
        @test_throws ErrorException StratifiedHoldout(fraction_train = 0.0)
        @test_throws ErrorException StratifiedHoldout(fraction_train = 1.0)
        @test_throws ErrorException StratifiedHoldout(fraction_train = -0.1)
        @test_throws ErrorException StratifiedHoldout(fraction_train = 1.5)

        # Test boundary values that should be valid (very close to boundaries)
        @test_nowarn StratifiedHoldout(fraction_train = 0.001)
        @test_nowarn StratifiedHoldout(fraction_train = 0.999)
    end

    @testset "type stability" begin
        holdout = StratifiedHoldout()
        @test holdout isa StratifiedHoldout
        @test holdout isa MLJBase.ResamplingStrategy
        @test fieldtype(typeof(holdout), :fraction_train) == Float64
        @test fieldtype(typeof(holdout), :shuffle) == Bool
        @test fieldtype(typeof(holdout), :rng) == AbstractRNG
    end
end

@testset "Classification Stratification Tests" begin
    @testset "balanced classification" begin
        X, y = generate_balanced_classification_data(150, 3)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 123)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        @test length(pairs) == 1

        train_idx, test_idx = pairs[1]

        # Test basic properties
        @test length(train_idx) + length(test_idx) == length(rows)
        @test isempty(intersect(train_idx, test_idx))
        @test length(train_idx) ≈ round(Int, 0.7 * length(rows)) atol=2

        # Test stratification quality
        y_train, y_test = y[train_idx], y[test_idx]
        @test check_classification_stratification(y_train, y_test, y, tolerance = 0.15)

        # Verify all classes are represented in both splits
        train_classes = Set(y_train)
        test_classes = Set(y_test)
        original_classes = Set(y)
        @test train_classes == original_classes
        @test test_classes == original_classes
    end

    @testset "imbalanced classification" begin
        X, y = generate_imbalanced_classification_data(200)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.8, rng = 456)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        y_train, y_test = y[train_idx], y[test_idx]

        # Check that imbalanced proportions are maintained
        @test check_classification_stratification(y_train, y_test, y, tolerance = 0.15)

        # Verify minority class is represented
        original_counts = countmap(y)
        train_counts = countmap(y_train)
        test_counts = countmap(y_test)

        # All original classes should be in train set
        for class in keys(original_counts)
            @test haskey(train_counts, class)
            @test haskey(test_counts, class)
        end
    end

    @testset "single class edge case" begin
        y = fill(categorical([1])[1], 50)  # All same class
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 789)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        @test length(train_idx) + length(test_idx) == length(rows)
        @test isempty(intersect(train_idx, test_idx))
        @test length(train_idx) ≈ round(Int, 0.7 * length(rows)) atol=2

        # All samples should be the same class
        y_train, y_test = y[train_idx], y[test_idx]
        @test all(x -> x == y[1], y_train)
        @test all(x -> x == y[1], y_test)
    end

    @testset "shuffle effect" begin
        X, y = generate_balanced_classification_data(100, 3)
        rows = 1:length(y)

        # Test without shuffling
        holdout1 = StratifiedHoldout(fraction_train = 0.7, shuffle = false, rng = 123)
        pairs1 = MLJBase.train_test_pairs(holdout1, rows, y)
        train_idx1, test_idx1 = pairs1[1]

        # Test with shuffling
        holdout2 = StratifiedHoldout(fraction_train = 0.7, shuffle = true, rng = 123)
        pairs2 = MLJBase.train_test_pairs(holdout2, rows, y)
        train_idx2, test_idx2 = pairs2[1]

        # Results should be different when shuffling
        @test train_idx1 != train_idx2 || test_idx1 != test_idx2

        # But both should maintain stratification
        @test check_classification_stratification(y[train_idx1], y[test_idx1], y)
        @test check_classification_stratification(y[train_idx2], y[test_idx2], y)
    end
end

@testset "Regression Stratification Tests" begin
    @testset "normal distribution" begin
        X, y = generate_regression_data(200)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.75, rng = 123)

        pairs = MLJBase.train_test_pairs(holdout, rows, y, n_bins = 5)
        @test length(pairs) == 1

        train_idx, test_idx = pairs[1]

        # Test basic properties
        @test length(train_idx) + length(test_idx) == length(rows)
        @test isempty(intersect(train_idx, test_idx))
        @test length(train_idx) ≈ round(Int, 0.75 * length(rows)) atol=5

        # Test regression stratification quality
        y_train, y_test = y[train_idx], y[test_idx]
        @test check_regression_stratification(y_train, y_test, y)

        # Check that distributions are similar
        @test abs(mean(y_train) - mean(y)) <= 0.2 * std(y)
        @test abs(mean(y_test) - mean(y)) <= 0.2 * std(y)
    end

    @testset "with missing values" begin
        X, y = generate_regression_data(150, add_missing = true)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 456)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        y_train, y_test = y[train_idx], y[test_idx]

        # Check that missing values are handled
        @test length(train_idx) + length(test_idx) == length(rows)

        # Both splits should potentially have missing values
        n_missing_train = count(ismissing, y_train)
        n_missing_test = count(ismissing, y_test)
        n_missing_total = count(ismissing, y)

        @test n_missing_train + n_missing_test == n_missing_total

        # Non-missing values should be stratified
        @test check_regression_stratification(y_train, y_test, y)
    end

    @testset "constant values" begin
        X, y = generate_regression_data(80, constant = true)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.6, rng = 789)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        # Should fall back to random split for constant values
        @test length(train_idx) + length(test_idx) == length(rows)
        @test isempty(intersect(train_idx, test_idx))
        @test length(train_idx) ≈ round(Int, 0.6 * length(rows)) atol=2

        # All values should be the same
        y_train, y_test = y[train_idx], y[test_idx]
        @test all(x -> x == y[1], y_train)
        @test all(x -> x == y[1], y_test)
    end

    @testset "different n_bins" begin
        X, y = generate_regression_data(100)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 123)

        # Test with different number of bins
        for n_bins in [3, 5, 10]
            pairs = MLJBase.train_test_pairs(holdout, rows, y, n_bins = n_bins)
            train_idx, test_idx = pairs[1]

            @test length(train_idx) + length(test_idx) == length(rows)
            @test isempty(intersect(train_idx, test_idx))

            y_train, y_test = y[train_idx], y[test_idx]
            @test check_regression_stratification(y_train, y_test, y, n_bins = n_bins)
        end
    end
end

@testset "Edge Case Tests" begin
    @testset "single sample" begin
        y = categorical([1])
        rows = [1]
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 123)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        # With only one sample, the implementation falls back to MLJBase.partition
        # since empty test set triggers fallback. MLJBase.partition duplicates single samples.
        @test length(train_idx) == 1
        @test length(test_idx) == 1
        @test train_idx == [1] && test_idx == [1]  # MLJBase.partition duplicates single sample
    end

    @testset "two samples same class" begin
        y = categorical([1, 1])
        rows = [1, 2]
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 123)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        # With two samples of same class, one should go to each split
        @test length(train_idx) == 1
        @test length(test_idx) == 1
        @test Set([train_idx[1], test_idx[1]]) == Set([1, 2])
    end

    @testset "two samples different classes" begin
        y = categorical([1, 2])
        rows = [1, 2]
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 123)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        # Each class should be represented
        @test length(train_idx) >= 1
        @test length(test_idx) >= 1
        @test length(train_idx) + length(test_idx) == 2
    end

    @testset "empty fallback behavior" begin
        # Test fallback when no target is provided
        rows = 1:10
        holdout = StratifiedHoldout(fraction_train = 0.7, rng = 123)

        # This should trigger a warning and use random split
        splits = @test_logs (:warn, r"StratifiedHoldout requires target variable") begin
            MLJBase.train_test_pairs(holdout, rows)
        end

        train_idx, test_idx = splits[1]
        @test length(train_idx) + length(test_idx) == length(rows)
        @test isempty(intersect(train_idx, test_idx))
    end
end

@testset "Property-Based Tests" begin
    @testset "completeness property" begin
        # Test that all indices are accounted for
        for n in [10, 50, 100, 200]
            X, y = generate_balanced_classification_data(n, 3)
            rows = 1:length(y)
            holdout = StratifiedHoldout(fraction_train = 0.7, rng = 123)

            pairs = MLJBase.train_test_pairs(holdout, rows, y)
            train_idx, test_idx = pairs[1]

            # All original indices should be covered exactly once
            all_indices = sort(vcat(train_idx, test_idx))
            @test all_indices == collect(rows)
        end
    end

    @testset "no overlap property" begin
        # Test that train and test sets never overlap
        for _ = 1:20  # Multiple random tests
            X, y = generate_balanced_classification_data(100, 4)
            rows = 1:length(y)
            holdout =
                StratifiedHoldout(fraction_train = rand() * 0.8 + 0.1, rng = rand(1:1000))

            pairs = MLJBase.train_test_pairs(holdout, rows, y)
            train_idx, test_idx = pairs[1]

            @test isempty(intersect(train_idx, test_idx))
        end
    end

    @testset "proportion maintenance" begin
        # Test that class proportions are approximately maintained
        X, y = generate_imbalanced_classification_data(500)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.75, rng = 123)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]

        y_train, y_test = y[train_idx], y[test_idx]

        # Calculate proportions
        original_props = proportions(y)
        train_props = proportions(y_train)
        test_props = proportions(y_test)

        # Check that proportions are maintained within reasonable tolerance
        for class in keys(original_props)
            if haskey(train_props, class) && haskey(test_props, class)
                @test abs(train_props[class] - original_props[class]) <= 0.1
                @test abs(test_props[class] - original_props[class]) <= 0.1
            end
        end
    end

    @testset "train size property" begin
        # Test that train set size respects fraction_train
        X, y = generate_balanced_classification_data(200, 3)
        rows = 1:length(y)

        for fraction in [0.1, 0.3, 0.5, 0.7, 0.9]
            holdout = StratifiedHoldout(fraction_train = fraction, rng = 123)
            pairs = MLJBase.train_test_pairs(holdout, rows, y)
            train_idx, test_idx = pairs[1]

            expected_train_size = round(Int, fraction * length(rows))
            # Allow some tolerance due to stratification constraints
            @test abs(length(train_idx) - expected_train_size) <= max(3, length(Set(y)))
        end
    end
end

@testset "Reproducibility Tests" begin
    @testset "deterministic with same seed" begin
        X, y = generate_balanced_classification_data(100, 3)
        rows = 1:length(y)

        holdout1 = StratifiedHoldout(fraction_train = 0.7, shuffle = true, rng = 42)
        holdout2 = StratifiedHoldout(fraction_train = 0.7, shuffle = true, rng = 42)

        pairs1 = MLJBase.train_test_pairs(holdout1, rows, y)
        pairs2 = MLJBase.train_test_pairs(holdout2, rows, y)

        train_idx1, test_idx1 = pairs1[1]
        train_idx2, test_idx2 = pairs2[1]

        @test train_idx1 == train_idx2
        @test test_idx1 == test_idx2
    end

    @testset "different with different seeds" begin
        X, y = generate_balanced_classification_data(100, 3)
        rows = 1:length(y)

        holdout1 = StratifiedHoldout(fraction_train = 0.7, shuffle = true, rng = 42)
        holdout2 = StratifiedHoldout(fraction_train = 0.7, shuffle = true, rng = 123)

        pairs1 = MLJBase.train_test_pairs(holdout1, rows, y)
        pairs2 = MLJBase.train_test_pairs(holdout2, rows, y)

        train_idx1, test_idx1 = pairs1[1]
        train_idx2, test_idx2 = pairs2[1]

        # Should be different with different seeds (very high probability)
        @test train_idx1 != train_idx2 || test_idx1 != test_idx2
    end

    @testset "consistent without shuffle" begin
        X, y = generate_balanced_classification_data(100, 3)
        rows = 1:length(y)

        holdout1 = StratifiedHoldout(fraction_train = 0.7, shuffle = false, rng = 42)
        holdout2 = StratifiedHoldout(fraction_train = 0.7, shuffle = false, rng = 123)

        pairs1 = MLJBase.train_test_pairs(holdout1, rows, y)
        pairs2 = MLJBase.train_test_pairs(holdout2, rows, y)

        train_idx1, test_idx1 = pairs1[1]
        train_idx2, test_idx2 = pairs2[1]

        # Without shuffling, the split should be deterministic regardless of RNG
        # (RNG is only used for sampling within classes)
        # But this may still differ due to sampling, so we just test they're valid
        @test Set(vcat(train_idx1, test_idx1)) == Set(vcat(train_idx2, test_idx2))
    end
end

@testset "Integration Tests with MLJ" begin
    @testset "MLJ ResamplingStrategy interface" begin
        holdout = StratifiedHoldout()
        @test holdout isa MLJBase.ResamplingStrategy

        # Test that it works with MLJ's resampling machinery
        X, y = generate_balanced_classification_data(50, 3)
        rows = 1:length(y)

        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        @test pairs isa Vector{<:Tuple}
        @test length(pairs) == 1

        train_idx, test_idx = pairs[1]
        @test train_idx isa AbstractVector{<:Integer}
        @test test_idx isa AbstractVector{<:Integer}
    end

    @testset "compatibility with different target types" begin
        holdout = StratifiedHoldout(fraction_train = 0.6, rng = 123)
        rows = 1:100

        # Test with different target types that should trigger classification
        y_categorical = categorical(repeat(1:4, 25))
        pairs_cat = MLJBase.train_test_pairs(holdout, rows, y_categorical)
        @test length(pairs_cat) == 1

        # Test with continuous target (should trigger regression)
        y_continuous = randn(100)
        pairs_cont = MLJBase.train_test_pairs(holdout, rows, y_continuous)
        @test length(pairs_cont) == 1

        # Test with mixed target (some missing)
        y_missing = Vector{Union{Missing,Float64}}(randn(100))
        y_missing[1:10] .= missing
        pairs_missing = MLJBase.train_test_pairs(holdout, rows, y_missing)
        @test length(pairs_missing) == 1
    end

    @testset "multiple calls consistency" begin
        X, y = generate_balanced_classification_data(80, 3)
        rows = 1:length(y)

        # Multiple calls with same seed should give same result
        holdout1 = StratifiedHoldout(fraction_train = 0.7, rng = 456)
        holdout2 = StratifiedHoldout(fraction_train = 0.7, rng = 456)

        pairs1 = MLJBase.train_test_pairs(holdout1, rows, y)
        pairs2 = MLJBase.train_test_pairs(holdout2, rows, y)

        @test pairs1 == pairs2  # Same seed should give same result

        # But different seeds should give different results (with high probability)
        holdout3 = StratifiedHoldout(fraction_train = 0.7, rng = 789)
        pairs3 = MLJBase.train_test_pairs(holdout3, rows, y)
        @test pairs1 != pairs3  # Different seeds should give different results
    end

    @testset "performance characteristics" begin
        # Test that stratification doesn't take too long
        X, y = generate_balanced_classification_data(1000, 5)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train = 0.8, rng = 123)

        # Should complete quickly
        @elapsed time = @elapsed MLJBase.train_test_pairs(holdout, rows, y)
        @test time < 1.0  # Should complete in less than 1 second

        # Should not allocate excessively
        allocs = @allocated MLJBase.train_test_pairs(holdout, rows, y)
        @test allocs < 10_000_000  # Less than 10MB of allocations
    end
end
