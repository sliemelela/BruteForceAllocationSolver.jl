using BruteForceAllocationSolver
using Test

@testset "Merton Benchmark Test" begin

    # Define amount of timesteps, stepsize etc.
    M = 10
    dt = 1.0
    β = 0.96

    # Define utility function
    γ = 5.0
    u(x) = (x^(1 - γ))/(1 - γ)

    # Setup grid parameters
    G_w, W_min, W_max = 500, 1.0, 100.0
    G_c, c_min, c_max = 50, 0.01, 0.99
    G_ω, ω_min, ω_max = 101, 0.0, 1.0

    # Setup grids
    W_grid = generate_log_spaced_grid(W_min, W_max, G_w)
    Z_grids = Vector{Float64}[]
    c_grid = collect(range(c_min, c_max, length=G_c))
    omega_space = [[ω] for ω in range(ω_min, ω_max, length=G_ω)]

    # Generate 1D integration nodes using the new automated function
    ρ_mat = fill(1.0, 1, 1) # 1x1 correlation matrix for 1D
    ε_nodes, W_weights = generate_gaussian_shocks(1, 10, ρ_mat)

    # Dynamics of the state variables
    r = 0.02
    μ = 0.07
    σ = 0.2
    merton_transition = make_merton_transition(r, μ, σ, dt)

    # Inject the extrapolator
    crra_extrapolator = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

    # Run the solver
    V, pol_c, pol_w = solve_dynamic_program(
        W_grid, Z_grids, c_grid, omega_space,
        ε_nodes, W_weights, merton_transition,
        M, β, u, fractional_consumption,
        standard_budget_constraint, crra_extrapolator
    )

    # Find analytical solution
    analytical_w = (μ - r) / (γ * σ^2) # 0.25

    # Check if the middle of the grid matches the analytical solution
    for W_idx in range(1, G_w), n in range(1, M)
        numerical_w = pol_w[W_idx, n][1]
        @test isapprox(numerical_w, analytical_w, atol=0.02)
    end
end

@testset "Merton Log-Wealth Benchmark Test" begin
    # 1. Setup Parameters
    γ = 5.0
    u(X) = (X^(1 - γ)) / (1 - γ)

    # 2. The Log-Space Grid
    G_X = 500
    X_grid = collect(range(log(0.01), log(100.0), length=G_X))
    Z_grids = Vector{Float64}[]
    M = 10
    β = 0.96
    c_grid = collect(range(0.01, 0.99, length=50))
    omega_space = [[w] for w in range(0.0, 1.0, length=101)]

    # 3. Generate 1D integration nodes
    ρ_mat = fill(1.0, 1, 1)
    ε_nodes, X_weights = generate_gaussian_shocks(1, 5, ρ_mat)

    # 4. Market Dynamics
    merton_transition = make_merton_transition(0.02, 0.07, 0.20, 1.0)

    # 5. Inject the Log-Space Strategies
    log_extrapolator = make_log_crra_extrapolator(X_grid[1], X_grid[end], γ)

    # 6. Run the solver!
    V, pol_c, pol_w = solve_dynamic_program(
        X_grid, Z_grids, c_grid, omega_space,
        ε_nodes, X_weights, merton_transition,
        M, β, u, log_fractional_consumption,
        log_budget_constraint, log_extrapolator
    )

    # 7. Validate
    analytical_w = (0.07 - 0.02) / (5.0 * 0.20^2) # 0.25

    for X_idx in range(1, G_X), n in range(1, M)
        numerical_w = pol_w[X_idx, n][1]
        @test isapprox(numerical_w, analytical_w, atol=0.02)
    end
end


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