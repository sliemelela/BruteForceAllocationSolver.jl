"""
    evaluate_bellman_objective(
        W_n::Float64, c_n::Float64, ω_n,
        shocks_precalc, W_weights::Vector{Float64},
        V_next::VType, β::Float64, u::UFunc,
        compute_consumption::CCFunc, budget_constraint::BFunc, extrapolator::EFunc
    )

Evaluates the objective function using strictly typed precalculated market transitions.
This function is called inside the control grid search and is allocation-free.

# Arguments
- `W_n::Float64`: The current principal state variable (e.g., Wealth or Log-Wealth).
- `c_n::Float64`: The chosen consumption control.
- `ω_n`: The chosen portfolio weight vector.
- `shocks_precalc`: A strictly typed collection of Tuple(Z_next, R_e, R_base) for each quadrature node.
- `W_weights::Vector{Float64}`: The quadrature weights corresponding to each node.
- `V_next`: The multidimensional interpolation object for next period's value.
- `β::Float64`: The subjective discount factor.
- `u`: The pure utility function.
- `compute_consumption`: Strategy to convert control to physical consumption.
- `budget_constraint`: Strategy for wealth evolution.
- `extrapolator`: Strategy for boundary handling.

# Returns
- `Float64`: The total objective value for the chosen controls.
"""
function evaluate_bellman_objective(
    W_n::Float64, c_n::Float64, ω_n,
    shocks_precalc::AbstractVector, W_weights::Vector{Float64},
    V_next::VType, β::Float64, u::UFunc,
    compute_consumption::CCFunc, budget_constraint::BFunc, extrapolator::EFunc
) where {VType, UFunc, CCFunc, BFunc, EFunc}

    expected_future_value = 0.0

    # Hot loop: Now strictly typed and allocation-free
    @inbounds for j in 1:length(W_weights)
        weight = W_weights[j]

        # Typed deconstruction from the precalculated collection
        Z_next, R_e, R_base = shocks_precalc[j]

        # 1. Wealth Transition
        W_next = budget_constraint(W_n, c_n, ω_n, R_e, R_base)

        # 2. Value Evaluation
        expected_future_value += weight * extrapolator(W_next, Z_next, V_next)
    end

    absolute_consumed = compute_consumption(W_n, c_n)
    current_utility = c_n > 0.0 ? u(absolute_consumed) : -Inf
    return current_utility + β * expected_future_value
end

function evaluate_bellman_objective(
    W_n::Float64, ω_n,
    shocks_precalc::AbstractVector, W_weights::Vector{Float64},
    V_next::VType, budget_constraint::BFunc, extrapolator::EFunc
) where {VType, BFunc, EFunc}

    expected_future_value = 0.0

    @inbounds for j in 1:length(W_weights)
        weight = W_weights[j]
        Z_next, R_e, R_base = shocks_precalc[j]

        # Wealth Transition (Injecting c_n = 0.0)
        W_next = budget_constraint(W_n, 0.0, ω_n, R_e, R_base)

        # Value Evaluation
        expected_future_value += weight * extrapolator(W_next, Z_next, V_next)
    end

    return expected_future_value
end

"""
    optimize_controls_brute_force(...)

Finds the optimal controls for a specific state (W_n, Z_n) using an exhaustive search.
Market transitions are precalculated exactly once before entering the control loops.
"""
function optimize_controls_brute_force(
    W_n::Float64, Z_n, c_grid::Vector{Float64}, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    V_next::VType, transition_model::TMod, β::Float64, u::UFunc,
    compute_consumption::CCFunc, budget_constraint::BFunc,
    extrapolator::EFunc
) where {VType, TMod, UFunc, CCFunc, BFunc, EFunc}

    # STEP 1: Strictly typed precalculation using an array comprehension.
    # This calls the expensive transition math (exponentials, etc.) ONLY once per node.
    shocks_precalc = [transition_model(Z_n, ε) for ε in ε_nodes]

    # STEP 2: Control Space Search (Millions of evaluations, now purely analytical)
    best_val = -Inf
    best_c = 0.0
    best_ω = first(omega_space)

    for c_n in c_grid, ω_n in omega_space
        current_val = evaluate_bellman_objective(
            W_n, c_n, ω_n,
            shocks_precalc, W_weights,
            V_next, β, u,
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

function optimize_controls_brute_force(
    W_n::Float64, Z_n, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    V_next::VType, transition_model::TMod,
    budget_constraint::BFunc, extrapolator::EFunc
) where {VType, TMod, BFunc, EFunc}

    shocks_precalc = [transition_model(Z_n, ε) for ε in ε_nodes]

    best_val = -Inf
    best_ω = first(omega_space)

    for ω_n in omega_space
        current_val = evaluate_bellman_objective(
            W_n, ω_n,
            shocks_precalc, W_weights,
            V_next, budget_constraint, extrapolator
        )

        if current_val > best_val
            best_val = current_val
            best_ω = ω_n
        end
    end

    return best_val, best_ω
end

"""
    solve_dynamic_program(...)

Solves a finite-horizon dynamic programming problem for portfolio and consumption choice
using backwards recursion and brute-force grid search.
"""
function solve_dynamic_program(
    solver::BruteForceSolver,
    W_grid::Vector{Float64}, Z_grids::Vector{Vector{Float64}},
    c_grid::Vector{Float64}, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    transition_model::Function, M::Int, β::Float64, u::Function,
    compute_consumption::Function, budget_constraint::Function, extrapolator::Function
)
    sz = (length(W_grid), (length(z) for z in Z_grids)...)
    V     = zeros(Float64, sz..., M + 1)
    pol_c = zeros(Float64, sz..., M)
    pol_w = Array{eltype(omega_space)}(undef, sz..., M)

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

            # Extract state coordinates as an SVector
            Z_n = ntuple(k -> Z_grids[k][idx[1+k]], length(Z_grids))

            best_val, best_c, best_ω = optimize_controls_brute_force(
                W_n, SVector(Z_n), c_grid, omega_space, ε_nodes, W_weights,
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

function solve_dynamic_program(
    solver::BruteForceSolver,
    W_grid::Vector{Float64}, Z_grids::Vector{Vector{Float64}},
    omega_space,
    ε_nodes, W_weights::Vector{Float64},
    transition_model::Function, M::Int, u::Function,
    state_to_wealth::Function, budget_constraint::Function, extrapolator::Function
)
    sz = (length(W_grid), (length(z) for z in Z_grids)...)
    V     = zeros(Float64, sz..., M + 1)
    pol_w = Array{eltype(omega_space)}(undef, sz..., M)

    println("Setting terminal conditions (Terminal Wealth)...")
    for idx in CartesianIndices(sz)
        state_terminal = W_grid[idx[1]]
        actual_wealth = state_to_wealth(state_terminal)
        V[idx, M+1] = u(actual_wealth)
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
            Z_n = ntuple(k -> Z_grids[k][idx[1+k]], length(Z_grids))

            best_val, best_ω = optimize_controls_brute_force(
                W_n, SVector(Z_n), omega_space, ε_nodes, W_weights,
                V_next_interp, transition_model,
                budget_constraint, extrapolator
            )

            V[idx, n]     = best_val
            pol_w[idx, n] = best_ω
        end
    end

    println("Recursion complete.")
    return V, pol_w
end