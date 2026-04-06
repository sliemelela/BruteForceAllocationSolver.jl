@testset "Terminal Wealth Overloads (No Consumption)" begin
    M = 10
    dt = 1.0
    γ = 5.0
    u(W) = (W^(1 - γ)) / (1 - γ)
    inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

    G_w = 200
    W_grid = generate_log_spaced_grid(1.0, 100.0, G_w)
    Z_grids = Vector{Float64}[]
    omega_space = [SVector(w) for w in range(0.0, 1.0, length=101)]

    ρ_mat = fill(1.0, 1, 1)
    ε_nodes, W_weights = generate_gaussian_shocks(1, 10, ρ_mat)

    r = 0.02
    μ = 0.07
    σ = 0.20
    merton_transition = make_merton_transition(r, μ, σ, dt)

    # Update to the new CE-based extrapolator (no gamma needed)
    ce_ex = make_ce_crra_extrapolator(W_grid[1], W_grid[end])

    # Rename V_term -> CE_term and inject inv_u
    CE_term, pol_w_term = solve_dynamic_program(
        BruteForceSolver(),
        W_grid, Z_grids, omega_space,
        ε_nodes, W_weights, merton_transition,
        M, u, inv_u, identity, standard_budget_constraint, ce_ex
    )

    @test size(CE_term) == (G_w, M + 1)
    @test size(pol_w_term) == (G_w, M)
    @test !@isdefined(pol_c)

    # Analytical Economic Solutions
    analytical_w = (μ - r) / (γ * σ^2)
    certainty_growth_rate = r + (μ - r)^2 / (2.0 * γ * σ^2) # The closed-form CE rate

    middle_idx = div(G_w, 2)
    for n in 1:M
        # 1. Test Portfolio Weight
        numerical_w = pol_w_term[middle_idx, n][1]
        @test isapprox(numerical_w, analytical_w, atol=0.02)

        # 2. Test Certainty Equivalent
        time_to_maturity = (M - n + 1) * dt
        W_current = W_grid[middle_idx]

        analytical_CE = W_current * exp(certainty_growth_rate * time_to_maturity)

        # 3. No need to calculate CE manually; it is natively stored in the grid
        numerical_CE = CE_term[middle_idx, n]

        # We use a 5% relative tolerance (rtol) to account for Euler discretization error
        @test isapprox(numerical_CE, analytical_CE, rtol=0.05)
    end
end