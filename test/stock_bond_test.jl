@testset "Rolling Nominal Bond and Stock (2 Risky Assets)" begin

    M, dt, γ = 1, 0.5, 5.0
    u(W) = (W^(1 - γ)) / (1 - γ)

    G_X = 300
    X_grid = generate_linear_grid(log(0.01), log(100.0), G_X)
    G_r = 3
    Z_grids = [generate_linear_grid(0.0, 0.10, G_r)]

    w_bond_space = range(0.20, 0.45, length=26)
    w_stock_space = range(0.15, 0.35, length=21)
    omega_space = vec([[wb, ws] for wb in w_bond_space, ws in w_stock_space])

    ρ_val = 0.0
    ρ_mat = [1.0 ρ_val; ρ_val 1.0]
    ε_nodes, W_weights = generate_gaussian_shocks(2, 8, ρ_mat)
    log_extrapolator = make_log_crra_extrapolator(X_grid[1], X_grid[end], γ)

    κ, θ, σ_r = 0.1, 0.03, 0.01
    λ_r, τ = -0.1, 10.0
    λ_S, σ_S = 0.25, 0.20

    transition_bond_stock = make_stochastic_r_bond_stock_transition(
        κ, θ, σ_r, λ_r, τ, λ_S, σ_S, ρ_val, dt
    )

    _, pol_w = solve_dynamic_program(
        X_grid, Z_grids, omega_space,
        ε_nodes, W_weights, transition_bond_stock,
        M, u, exp, log_budget_constraint, log_extrapolator # <--- Passed 'exp' here
    )

    B_r = (1.0 - exp(-κ * τ)) / κ
    analytical_w_stock = (λ_S * σ_S) / (γ * σ_S^2)
    analytical_w_bond  = (-λ_r * B_r * σ_r) / (γ * (B_r * σ_r)^2)

    num_w_bond  = pol_w[5, 2, 1][1]
    num_w_stock = pol_w[5, 2, 1][2]

    @test isapprox(num_w_stock, analytical_w_stock, atol=0.02)
    @test isapprox(num_w_bond, analytical_w_bond, atol=0.02)
end