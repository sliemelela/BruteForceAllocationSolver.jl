
@testset "Stochastic Interest Rate Models (2D Quadrature)" begin
    # 1. Base Parameters
    M, dt, β, γ = 5, 1.0, 0.96, 5.0
    u(X) = (X^(1 - γ)) / (1 - γ)

    # 2. Set up Grids
    G_X = 100
    X_grid = generate_linear_grid(log(0.01), log(100.0), G_X)

    # Auxiliary state variable Z (The interest rate r)
    G_r = 11
    Z_grids = [generate_linear_grid(0.0, 0.10, G_r)]

    c_grid = generate_linear_grid(0.01, 0.99, 50)
    omega_space = [[w] for w in generate_linear_grid(0.0, 1.0, 101)]

    # 3. 2D Quadrature Integration Setup
    ρ_val = 0.0 # Correlation between interest rate and stock shocks
    ρ_mat = [1.0 ρ_val; ρ_val 1.0] # 2x2 Correlation Matrix

    # Generate 2D integration nodes automatically
    ε_nodes, W_weights = generate_gaussian_shocks(2, 10, ρ_mat)

    # 4. Shared Economic Strategies
    log_extrapolator = make_log_crra_extrapolator(X_grid[1], X_grid[end], γ)

    # =========================================================================
    # Test Case A: Constant Mu (Optimal weight should SHRINK as r rises)
    # =========================================================================
    κ, θ, σ_r = 0.1, 0.03, 0.01
    μ, σ_S = 0.07, 0.20

    transition_mu = make_stochastic_r_constant_mu_transition(
        κ, θ, σ_r, μ, σ_S, ρ_val, dt
    )

    _, _, pol_w_mu = solve_dynamic_program(
        X_grid, Z_grids, c_grid, omega_space,
        ε_nodes, W_weights, transition_mu,
        M, β, u, log_fractional_consumption,
        log_budget_constraint, log_extrapolator
    )

    # Validate
    weight_at_r_0  = pol_w_mu[50, 1, 1][1]  # r = 0.0%
    weight_at_r_10 = pol_w_mu[50, 11, 1][1] # r = 10.0%

    @test weight_at_r_0 > weight_at_r_10

    # Specifically, at r = 0.02, it should perfectly match the Merton share
    analytical_merton_02 = (0.07 - 0.02) / (γ * σ_S^2) # = 0.25
    @test isapprox(pol_w_mu[50, 3, 1][1], analytical_merton_02, atol=0.03)

    # =========================================================================
    # Test Case B: Constant Risk Premium (Optimal weight should be FLAT)
    # =========================================================================
    λ_S = 0.25

    transition_premium = make_stochastic_r_constant_premium_transition(
        κ, θ, σ_r, λ_S, σ_S, ρ_val, dt
    )

    _, _, pol_w_premium = solve_dynamic_program(
        X_grid, Z_grids, c_grid, omega_space,
        ε_nodes, W_weights, transition_premium,
        M, β, u, log_fractional_consumption,
        log_budget_constraint, log_extrapolator
    )

    # Validate
    analytical_premium = (λ_S * σ_S) / (γ * σ_S^2) # = 0.05 / (5 * 0.04) = 0.25

    for r_idx in 1:G_r
        @test isapprox(pol_w_premium[50, r_idx, 1][1], analytical_premium, atol=0.03)
    end
end