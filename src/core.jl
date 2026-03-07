function evaluate_bellman_objective(
    W_n::Float64, Z_n::Vector{Float64}, c_n::Float64, ω_n::Vector{Float64},
    ε_nodes::Vector{Vector{Float64}}, W_weights::Vector{Float64},
    V_next, transition_model::Function, β::Float64, u::Function,
    compute_consumption::Function, budget_constraint::Function, extrapolator::Function
)
    expected_future_value = 0.0

    for j in 1:length(ε_nodes)
        ε_j = ε_nodes[j]
        weight = W_weights[j]

        # 1. Market Transitions
        Z_next, R_e, R_base = transition_model(Z_n, ε_j)

        # 2. Wealth Transition (Agnostic!)
        W_next = budget_constraint(W_n, c_n, ω_n, R_e, R_base)

        # 3. Value Evaluation (Agnostic!)
        expected_future_value += weight * extrapolator(W_next, Z_next, V_next)
    end

    absolute_consumed = compute_consumption(W_n, c_n)
    current_utility = c_n > 0.0 ? u(absolute_consumed) : -Inf
    return current_utility + β * expected_future_value
end

function optimize_controls_brute_force(
    W_n::Float64, Z_n::Vector{Float64}, c_grid::Vector{Float64}, omega_space,
    ε_nodes::Vector{Vector{Float64}}, W_weights::Vector{Float64},
    V_next, transition_model::Function, β::Float64, u::Function,
    compute_consumption::Function, budget_constraint::Function,
    extrapolator::Function
)
    best_val = -Inf
    best_c = 0.0
    best_ω = first(omega_space)

    for c_n in c_grid, ω_n in omega_space
        current_val = evaluate_bellman_objective(
            W_n, Z_n, c_n, ω_n, ε_nodes, W_weights,
            V_next, transition_model, β, u,
            compute_consumption, budget_constraint, extrapolator
        )

        if current_val > best_val
            best_val = current_val
            best_c = c_n
            best_ω = ω_n
        end
    end

    return best_val, best_c, best_ω
end

function solve_dynamic_program(
    W_grid::Vector{Float64}, Z_grids::Vector{Vector{Float64}},
    c_grid::Vector{Float64}, omega_space::Vector{Vector{Float64}},
    ε_nodes::Vector{Vector{Float64}}, W_weights::Vector{Float64},
    transition_model::Function, M::Int, β::Float64, u::Function,
    compute_consumption::Function, budget_constraint::Function, extrapolator::Function
)
    sz = (length(W_grid), (length(z) for z in Z_grids)...)
    V     = zeros(Float64, sz..., M + 1)
    pol_c = zeros(Float64, sz..., M)
    pol_w = Array{Vector{Float64}}(undef, sz..., M)

    println("Setting terminal conditions...")
    for idx in CartesianIndices(sz)
        state_terminal = W_grid[idx[1]]
        C_terminal = compute_consumption(state_terminal, 1.0)
        V[idx, M+1] = u(C_terminal)
    end

    println("Starting backwards recursion from step $M down to 1...")
    for n in M:-1:1
        println("  Solving timestep: $n")

        V_next_data = selectdim(V, ndims(V), n + 1)
        V_next_interp = linear_interpolation(
            (W_grid, Z_grids...), V_next_data, extrapolation_bc=Line()
        )

        Threads.@threads for idx in CartesianIndices(sz)
            W_n = W_grid[idx[1]]
            Z_n = [Z_grids[k][idx[1+k]] for k in 1:length(Z_grids)]

            best_val, best_c, best_ω = optimize_controls_brute_force(
                W_n, Z_n, c_grid, omega_space, ε_nodes, W_weights,
                V_next_interp, transition_model, β, u,
                compute_consumption, budget_constraint, extrapolator
            )

            V[idx, n]     = best_val
            pol_c[idx, n] = best_c
            pol_w[idx, n] = best_ω
        end
    end

    println("Recursion complete.")
    return V, pol_c, pol_w
end