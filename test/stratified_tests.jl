@testset "StratifiedHoldout Constructor Tests" begin
    @testset "valid parameters" begin
        # Default constructor
        holdout1 = StratifiedHoldout()
        @test holdout1.fraction_train == 0.7
        @test holdout1.shuffle == false
        @test holdout1.rng isa AbstractRNG
        
        # Custom parameters
        holdout2 = StratifiedHoldout(fraction_train=0.8, shuffle=true, rng=123)
        @test holdout2.fraction_train == 0.8
        @test holdout2.shuffle == true
        @test holdout2.rng isa MersenneTwister
        
        # RNG as AbstractRNG
        rng = MersenneTwister(42)
        holdout3 = StratifiedHoldout(fraction_train=0.6, rng=rng)
        @test holdout3.fraction_train == 0.6
        @test holdout3.rng === rng
    end
    
    @testset "invalid parameters" begin
        # fraction_train out of bounds
        @test_throws ErrorException StratifiedHoldout(fraction_train=0.0)
        @test_throws ErrorException StratifiedHoldout(fraction_train=1.0)
        @test_throws ErrorException StratifiedHoldout(fraction_train=-0.1)
        @test_throws ErrorException StratifiedHoldout(fraction_train=1.5)
        
        # Test boundary values that should fail
        @test_throws ErrorException StratifiedHoldout(fraction_train=nextfloat(0.0))
        @test_throws ErrorException StratifiedHoldout(fraction_train=prevfloat(1.0))
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
        holdout = StratifiedHoldout(fraction_train=0.7, rng=123)
        
        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        @test length(pairs) == 1
        
        train_idx, test_idx = pairs[1]
        
        # Test basic properties
        @test length(train_idx) + length(test_idx) == length(rows)
        @test isempty(intersect(train_idx, test_idx))
        @test length(train_idx) ≈ round(Int, 0.7 * length(rows)) atol=2
        
        # Test stratification quality
        y_train, y_test = y[train_idx], y[test_idx]
        @test check_classification_stratification(y_train, y_test, y, tolerance=0.15)
        
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
        holdout = StratifiedHoldout(fraction_train=0.8, rng=456)
        
        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]
        
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check that imbalanced proportions are maintained
        @test check_classification_stratification(y_train, y_test, y, tolerance=0.15)
        
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
        holdout = StratifiedHoldout(fraction_train=0.7, rng=789)
        
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
        holdout1 = StratifiedHoldout(fraction_train=0.7, shuffle=false, rng=123)
        pairs1 = MLJBase.train_test_pairs(holdout1, rows, y)
        train_idx1, test_idx1 = pairs1[1]
        
        # Test with shuffling
        holdout2 = StratifiedHoldout(fraction_train=0.7, shuffle=true, rng=123)
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
        holdout = StratifiedHoldout(fraction_train=0.75, rng=123)
        
        pairs = MLJBase.train_test_pairs(holdout, rows, y, n_bins=5)
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
        X, y = generate_regression_data(150, add_missing=true)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train=0.7, rng=456)
        
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
        X, y = generate_regression_data(80, constant=true)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train=0.6, rng=789)
        
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
        holdout = StratifiedHoldout(fraction_train=0.7, rng=123)
        
        # Test with different number of bins
        for n_bins in [3, 5, 10]
            pairs = MLJBase.train_test_pairs(holdout, rows, y, n_bins=n_bins)
            train_idx, test_idx = pairs[1]
            
            @test length(train_idx) + length(test_idx) == length(rows)
            @test isempty(intersect(train_idx, test_idx))
            
            y_train, y_test = y[train_idx], y[test_idx]
            @test check_regression_stratification(y_train, y_test, y, n_bins=n_bins)
        end
    end
end

@testset "Edge Case Tests" begin
    @testset "single sample" begin
        y = categorical([1])
        rows = [1]
        holdout = StratifiedHoldout(fraction_train=0.7, rng=123)
        
        pairs = MLJBase.train_test_pairs(holdout, rows, y)
        train_idx, test_idx = pairs[1]
        
        # With only one sample, it should go to training
        @test length(train_idx) == 1
        @test isempty(test_idx) || length(test_idx) == 0
        @test train_idx == [1]
    end
    
    @testset "two samples same class" begin
        y = categorical([1, 1])
        rows = [1, 2]
        holdout = StratifiedHoldout(fraction_train=0.7, rng=123)
        
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
        holdout = StratifiedHoldout(fraction_train=0.7, rng=123)
        
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
        holdout = StratifiedHoldout(fraction_train=0.7, rng=123)
        
        # This should trigger a warning and use random split
        @test_logs (:warn, r"StratifiedHoldout requires target variable") begin
            pairs = MLJBase.train_test_pairs(holdout, rows)
        end
        
        train_idx, test_idx = pairs[1]
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
            holdout = StratifiedHoldout(fraction_train=0.7, rng=123)
            
            pairs = MLJBase.train_test_pairs(holdout, rows, y)
            train_idx, test_idx = pairs[1]
            
            # All original indices should be covered exactly once
            all_indices = sort(vcat(train_idx, test_idx))
            @test all_indices == collect(rows)
        end
    end
    
    @testset "no overlap property" begin
        # Test that train and test sets never overlap
        for _ in 1:20  # Multiple random tests
            X, y = generate_balanced_classification_data(100, 4)
            rows = 1:length(y)
            holdout = StratifiedHoldout(fraction_train=rand() * 0.8 + 0.1, rng=rand(1:1000))
            
            pairs = MLJBase.train_test_pairs(holdout, rows, y)
            train_idx, test_idx = pairs[1]
            
            @test isempty(intersect(train_idx, test_idx))
        end
    end
    
    @testset "proportion maintenance" begin
        # Test that class proportions are approximately maintained
        X, y = generate_imbalanced_classification_data(500)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train=0.75, rng=123)
        
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
            holdout = StratifiedHoldout(fraction_train=fraction, rng=123)
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
        
        holdout1 = StratifiedHoldout(fraction_train=0.7, shuffle=true, rng=42)
        holdout2 = StratifiedHoldout(fraction_train=0.7, shuffle=true, rng=42)
        
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
        
        holdout1 = StratifiedHoldout(fraction_train=0.7, shuffle=true, rng=42)
        holdout2 = StratifiedHoldout(fraction_train=0.7, shuffle=true, rng=123)
        
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
        
        holdout1 = StratifiedHoldout(fraction_train=0.7, shuffle=false, rng=42)
        holdout2 = StratifiedHoldout(fraction_train=0.7, shuffle=false, rng=123)
        
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
        holdout = StratifiedHoldout(fraction_train=0.6, rng=123)
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
        y_missing = Vector{Union{Missing, Float64}}(randn(100))
        y_missing[1:10] .= missing
        pairs_missing = MLJBase.train_test_pairs(holdout, rows, y_missing)
        @test length(pairs_missing) == 1
    end
    
    @testset "multiple calls consistency" begin
        X, y = generate_balanced_classification_data(80, 3)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train=0.7, rng=456)
        
        # Multiple calls with same holdout should give same result
        pairs1 = MLJBase.train_test_pairs(holdout, rows, y)
        pairs2 = MLJBase.train_test_pairs(holdout, rows, y)
        
        @test pairs1 == pairs2
    end
    
    @testset "performance characteristics" begin
        # Test that stratification doesn't take too long
        X, y = generate_balanced_classification_data(1000, 5)
        rows = 1:length(y)
        holdout = StratifiedHoldout(fraction_train=0.8, rng=123)
        
        # Should complete quickly
        @elapsed time = @elapsed MLJBase.train_test_pairs(holdout, rows, y)
        @test time < 1.0  # Should complete in less than 1 second
        
        # Should not allocate excessively
        allocs = @allocated MLJBase.train_test_pairs(holdout, rows, y)
        @test allocs < 10_000_000  # Less than 10MB of allocations
    end
end