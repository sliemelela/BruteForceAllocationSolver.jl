
@testset "Terminal Wealth Overloads (No Consumption)" begin
    # 1. Setup Parameters
    M = 10
    dt = 1.0
    γ = 5.0
    u(W) = (W^(1 - γ)) / (1 - γ)

    # 2. Setup Grids
    G_w = 200
    W_grid = generate_log_spaced_grid(1.0, 100.0, G_w)
    Z_grids = Vector{Float64}[]

    # Notice: NO c_grid is defined!
    omega_space = [[w] for w in range(0.0, 1.0, length=101)]

    # 3. Integration Nodes
    ρ_mat = fill(1.0, 1, 1)
    ε_nodes, W_weights = generate_gaussian_shocks(1, 10, ρ_mat)

    # 4. Market Dynamics
    r = 0.02
    μ = 0.07
    σ = 0.20
    merton_transition = make_merton_transition(r, μ, σ, dt)
    crra_ex = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

    # 5. Run the new Overloaded Solver
    # Notice: c_grid, compute_consumption, and β are completely omitted!
    V_term, pol_w_term = solve_dynamic_program(
        W_grid, Z_grids, omega_space,
        ε_nodes, W_weights, merton_transition,
        M, u, standard_budget_constraint, crra_ex
    )

    # 6. Validate Output Structure
    # V should have M+1 steps, pol_w should have M steps
    @test size(V_term) == (G_w, M + 1)
    @test size(pol_w_term) == (G_w, M)

    # Check that it correctly dropped the unused consumption policy
    @test !@isdefined(pol_c)

    # 7. Validate Analytical Economic Solution
    # For a pure terminal wealth Merton problem, the optimal weight is constant
    analytical_w = (μ - r) / (γ * σ^2) # (0.07 - 0.02) / (5 * 0.04) = 0.25

    # Check the middle of the wealth grid across all timesteps
    # (Avoids deep edges where boundary extrapolation carries tiny numerical artifacts)
    middle_idx = div(G_w, 2)
    for n in 1:M
        numerical_w = pol_w_term[middle_idx, n][1]
        @test isapprox(numerical_w, analytical_w, atol=0.02)
    end
end
