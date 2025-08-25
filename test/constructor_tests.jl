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