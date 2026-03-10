using BruteForceAllocationSolver
using FastGaussQuadrature
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

    # Create nodes
    Q = 10
    ε_nodes, W_weights = gausshermite(Q, normalize=true)
    ε_nodes = [[n] for n in ε_nodes]

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

    # Utility function in wealth terms
    u(X) = (X^(1 - γ)) / (1 - γ)

    # 2. The Log-Space Grid
    # Instead of an exponential grid for W, we just use a linear grid for X!
    # log(0.01) ≈ -4.6, log(100.0) ≈ 4.6
    G_X = 500
    X_grid = collect(range(log(0.01), log(100.0), length=G_X))
    Z_grids = Vector{Float64}[]
    M = 10
    β = 0.96
    c_grid = collect(range(0.01, 0.99, length=50))
    omega_space = [[w] for w in range(0.0, 1.0, length=101)]

    # 3. Quadrature Setup
    nodes, weights = gausshermite(5)
    ε_nodes = [[n * sqrt(2.0)] for n in nodes]
    X_weights = weights ./ sqrt(pi)

    # 4. Market Dynamics (Remains identical!)
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