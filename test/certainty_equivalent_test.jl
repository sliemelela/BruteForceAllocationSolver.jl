@testset "Equivalent Consumption Stream Utility" begin
    # Define the CRRA utility function and its mathematical inverse
    γ = 5.0
    u(x) = (x^(1.0 - γ)) / (1.0 - γ)
    inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

    β = 0.96
    periods_remaining = 2
    C_target = 5.0

    # 1. Simulate the solver's native output (Lump-Sum CE)
    # The solver evaluates the lifetime utility of a flat stream and stores it as a CE.
    discount_sum = 1.0 + β + β^2
    V_stream = u(C_target) * discount_sum
    CE_lump_sum = inv_u(V_stream) # This is what the new solve_dynamic_program returns!

    # 2. Scalar Test
    cec_scalar = calculate_equivalent_consumption_stream(CE_lump_sum, u, inv_u, β, periods_remaining)
    @test isapprox(cec_scalar, C_target, atol=1e-6)

    # 3. Array Test
    C_array = [5.0, 15.0, 25.0]
    V_stream_array = u.(C_array) .* discount_sum
    CE_lump_sum_array = inv_u.(V_stream_array)

    cec_array = calculate_equivalent_consumption_stream(CE_lump_sum_array, u, inv_u, β, periods_remaining)
    @test all(isapprox.(cec_array, C_array, atol=1e-6))
end