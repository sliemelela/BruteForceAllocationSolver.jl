using Optim

@testset "Merton Benchmark Test" begin

    # Define amount of timesteps, stepsize etc.
    M = 10
    dt = 1.0
    β = 0.96

    # Define utility function and its mathematical inverse (CE)
    γ = 5.0
    u(x) = (x^(1 - γ))/(1 - γ)
    inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

    # Setup grid parameters
    G_w, W_min, W_max = 500, 1.0, 100.0
    G_c, c_min, c_max = 50, 0.01, 0.99
    G_ω, ω_min, ω_max = 101, 0.0, 1.0

    # Setup grids
    W_grid = generate_log_spaced_grid(W_min, W_max, G_w)
    Z_grids = Vector{Float64}[]
    c_grid = collect(range(c_min, c_max, length=G_c))
    omega_space = [SVector(ω) for ω in range(ω_min, ω_max, length=G_ω)]

    # Generate 1D integration nodes using the new automated function
    ρ_mat = fill(1.0, 1, 1) # 1x1 correlation matrix for 1D
    ε_nodes, W_weights = generate_gaussian_shocks(1, 10, ρ_mat)

    # Dynamics of the state variables
    r = 0.02
    μ = 0.07
    σ = 0.2
    merton_transition = make_merton_transition(r, μ, σ, dt)

    # <--- 2. UPDATED: Inject the CE extrapolator (no γ needed)
    ce_extrapolator = make_ce_crra_extrapolator(W_grid[1], W_grid[end])

    # Define solvers to test (using lighter parameters for Zooming to keep tests fast)
    solvers = [
        BruteForceSolver(),
        ZoomingSolver(iterations=2, points_per_dim=7, zoom_range_factor=1.5),
        OptimSolver(method=LBFGS(), use_gradients=true)
    ]

    analytical_w = (μ - r) / (γ * σ^2) # 0.25

    for solver in solvers
        @testset "$(typeof(solver))" begin
            # <--- 3. UPDATED: Run the solver with inv_u, and rename V to CE
            CE, pol_c, pol_w = solve_dynamic_program(
                solver,
                W_grid, Z_grids, c_grid, omega_space,
                ε_nodes, W_weights, merton_transition,
                M, β, u, inv_u, fractional_consumption, # <-- injected inv_u
                standard_budget_constraint, ce_extrapolator
            )

            # Check if the grid matches the analytical solution
            for W_idx in range(1, G_w), n in range(1, M)
                numerical_w = pol_w[W_idx, n][1]

                # The continuous solvers may achieve much higher precision than
                # the 0.01 step size of the brute force coarse grid, but we keep
                # the standard tolerance check to ensure baseline economic validity.
                @test isapprox(numerical_w, analytical_w, atol=0.02)
            end
        end
    end
end


@testset "Merton Log-Wealth Benchmark Test" begin
    # 1. Setup Parameters
    γ = 5.0
    u(X) = (X^(1 - γ)) / (1 - γ)
    inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ)) # <--- 1. NEW: Inverse Utility

    # 2. The Log-Space Grid
    G_X = 500
    X_grid = collect(range(log(0.01), log(100.0), length=G_X))
    Z_grids = Vector{Float64}[]
    M = 10
    β = 0.96
    c_grid = collect(range(0.01, 0.99, length=50))
    omega_space = [SVector(w) for w in range(0.0, 1.0, length=101)]

    # 3. Generate 1D integration nodes
    ρ_mat = fill(1.0, 1, 1)
    ε_nodes, X_weights = generate_gaussian_shocks(1, 5, ρ_mat)

    # 4. Market Dynamics
    merton_transition = make_merton_transition(0.02, 0.07, 0.20, 1.0)

    # 5. Inject the Log-Space Strategies (no γ needed)
    ce_log_extrapolator = make_ce_log_crra_extrapolator(X_grid[1], X_grid[end])

    solvers = [
        BruteForceSolver(),
        ZoomingSolver(iterations=2, points_per_dim=7, zoom_range_factor=1.5),
        OptimSolver(method=LBFGS(), use_gradients=true)
    ]

    analytical_w = (0.07 - 0.02) / (5.0 * 0.20^2) # 0.25

    for solver in solvers
        @testset "$(typeof(solver))" begin
            # 6. Run the solver! (Inject inv_u and rename V to CE_log)
            CE_log, pol_c, pol_w = solve_dynamic_program(
                solver,
                X_grid, Z_grids, c_grid, omega_space,
                ε_nodes, X_weights, merton_transition,
                M, β, u, inv_u, log_fractional_consumption, # <-- injected inv_u
                log_budget_constraint, ce_log_extrapolator
            )

            # 7. Validate
            for X_idx in range(1, G_X), n in range(1, M)
                numerical_w = pol_w[X_idx, n][1]
                @test isapprox(numerical_w, analytical_w, atol=0.02)
            end
        end
    end
end