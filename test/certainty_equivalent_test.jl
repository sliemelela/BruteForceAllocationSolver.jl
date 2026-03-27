@testset "Certainty Equivalent Utilities" begin
    # 1. Define the CRRA utility function and its mathematical inverse
    γ = 5.0
    u(x) = (x^(1.0 - γ)) / (1.0 - γ)
    inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

    # ====================================================================
    # Test 1: Terminal Wealth Certainty Equivalent (No discounting)
    # ====================================================================
    W_target = 10.0
    V_terminal = u(W_target)

    # Scalar Test
    ce_scalar = calculate_certainty_equivalent(V_terminal, inv_u)
    @test isapprox(ce_scalar, W_target, atol=1e-6)

    # Array Test
    W_array = [10.0, 20.0, 50.0]
    V_array = u.(W_array)
    ce_array = calculate_certainty_equivalent(V_array, inv_u)
    @test all(isapprox.(ce_array, W_array, atol=1e-6))


    # ====================================================================
    # Test 2: Equivalent Consumption Stream (With discounting)
    # ====================================================================
    β = 0.96
    periods_remaining = 2
    C_target = 5.0

    # Calculate the exact total value function if the investor consumed
    # exactly C_target for the current period and the next two periods.
    discount_sum = 1.0 + β + β^2
    V_stream = u(C_target) * discount_sum

    # Scalar Test
    cec_scalar = calculate_equivalent_consumption_stream(V_stream, inv_u, β, periods_remaining)
    @test isapprox(cec_scalar, C_target, atol=1e-6)

    # Array Test
    C_array = [5.0, 15.0, 25.0]
    V_stream_array = u.(C_array) .* discount_sum
    cec_array = calculate_equivalent_consumption_stream(V_stream_array, inv_u, β, periods_remaining)
    @test all(isapprox.(cec_array, C_array, atol=1e-6))
end