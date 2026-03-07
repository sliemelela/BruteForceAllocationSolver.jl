using BruteForceAllocationSolver
using FastGaussQuadrature
using Test

@testset "Merton Benchmark Test" begin
    # 1. Setup exactly like your run() script
    γ = 5.0
    u(x) = (x^(1 - γ))/(1 - γ)

    W_grid = exp.(collect(range(log(1.0), log(100.0), length=500)))
    Z_grids = Vector{Float64}[]
    c_grid = collect(range(0.01, 0.99, length=50))
    omega_space = [[w] for w in range(0.0, 1.0, length=101)]

    nodes, weights = gausshermite(10)
    ε_nodes = [[n * sqrt(2.0)] for n in nodes]
    W_weights = weights ./ sqrt(pi)

    merton_transition = make_merton_transition(0.02, 0.07, 0.20, 1.0)

    # 2. Inject the Strategies
    crra_extrapolator = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

    # 3. Run the solver
    V, pol_c, pol_w = solve_dynamic_program(
        W_grid, Z_grids, c_grid, omega_space,
        ε_nodes, W_weights, merton_transition,
        10, 0.96, u, fractional_consumption,
        standard_budget_constraint, crra_extrapolator
    )

    # 4. Automate the validation!
    analytical_w = (0.07 - 0.02) / (5.0 * 0.20^2) # 0.25

    # Check if the middle of the grid matches the analytical solution
    numerical_w = pol_w[50, 1][1]
    display(pol_w)

    @test isapprox(numerical_w, analytical_w, atol=0.02)
end

@testset "Merton Log-Wealth Benchmark Test" begin
    # 1. Setup Parameters
    γ = 5.0

    # Utility function in wealth terms
    u(X) = (X^(1 - γ)) / (1 - γ)

    # 2. The Log-Space Grid
    # Instead of an exponential grid for W, we just use a linear grid for X!
    # log(0.01) ≈ -4.6, log(100.0) ≈ 4.6
    X_grid = collect(range(log(0.01), log(100.0), length=500))
    Z_grids = Vector{Float64}[]

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
        10, 0.96, u, log_fractional_consumption,
        log_budget_constraint, log_extrapolator
    )

    # 7. Validate
    analytical_w = (0.07 - 0.02) / (5.0 * 0.20^2) # 0.25
    numerical_w = pol_w[250, 1][1]
    numerical_w_2 = pol_w[490, 1][1]

    display(pol_w)

    @test isapprox(numerical_w, analytical_w, atol=0.02)
    @test isapprox(numerical_w_2, analytical_w, atol=0.02)
end