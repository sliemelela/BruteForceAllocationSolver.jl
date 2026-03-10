
"""
    evaluate_bellman_objective(
        W_n::Float64, Z_n::Vector{Float64}, c_n::Float64, ω_n::Vector{Float64},
        ε_nodes::Vector{Vector{Float64}}, W_weights::Vector{Float64},
        V_next, transition_model::Function, β::Float64, u::Function,
        compute_consumption::Function, budget_constraint::Function, extrapolator::Function
    )

Evaluates the objective function of the Bellman equation for a specific state `(W_n, Z_n)`
and a specific set of controls `(c_n, ω_n)`.

This function computes the sum of the current utility and the discounted expected future value.
The expectation is approximated using numerical quadrature integration over the provided nodes.

# Arguments
- `W_n::Float64`: The current principal state variable (e.g., Wealth or Log-Wealth).
- `Z_n::Vector{Float64}`: The current vector of auxiliary state variables.
- `c_n::Float64`: The chosen consumption control (e.g., fraction of wealth or absolute amount).
- `ω_n::Vector{Float64}`: The chosen portfolio weight vector for risky assets.
- `ε_nodes::Vector{Vector{Float64}}`: The quadrature nodes representing standard normal shocks.
- `W_weights::Vector{Float64}`: The quadrature weights corresponding to each node.
- `V_next`: A callable multidimensional interpolation object representing the future
    value function ``V_{n+1}``.
- `transition_model::Function`: The market dynamics function.
    Expected signature: `(Z, ε) -> (Z_next, R_e, R_base)`.
- `β::Float64`: The subjective discount factor.
- `u::Function`: The pure utility function. Expected signature: `u(C_actual)`.
- `compute_consumption::Function`: Strategy to convert the state and control into absolute
    physical consumption (e.g.  `(W,c) -> c * W or (X, c) -> c * exp(X)`).
- `budget_constraint::Function`: Strategy dictating how the main state variable transitions
    Expected signature: `(W, c, ω, R_e, R_base) -> W_next`
    (cf. see [`standard_budget_constraint`](@ref))).
- `extrapolator::Function`: Strategy dictating how to evaluate `V_next` when future states
    fall outside the defined grid bounds (cf. see [`make_crra_extrapolator`](@ref)).

# Returns
- `Float64`: The total objective value for the chosen controls.
    Evaluates to `-Inf` if the consumption rule results in zero or negative physical consumption.
"""
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


"""
    optimize_controls_brute_force(
        W_n::Float64, Z_n::Vector{Float64}, c_grid::Vector{Float64}, omega_space,
        ε_nodes::Vector{Vector{Float64}}, W_weights::Vector{Float64},
        V_next, transition_model::Function, β::Float64, u::Function,
        compute_consumption::Function, budget_constraint::Function, extrapolator::Function
    )

Finds the optimal consumption and portfolio controls for a specific state `(W_n, Z_n)`
using an exhaustive grid search.

Iterates over every combination of consumption in `c_grid` and portfolio weights in `omega_space`,
evaluates the Bellman objective, and returns the combination that maximizes the value function.

# Arguments
- `c_grid::Vector{Float64}`: The discretized 1D grid of possible consumption choices.
- `omega_space`: An iterable collection of portfolio weight vectors.
    *(See [`evaluate_bellman_objective`](@ref) for the remaining arguments.)*

# Returns
A 3-tuple containing:
- `best_val::Float64`: The maximum value achieved.
- `best_c::Float64`: The optimal consumption choice.
- `best_ω::Vector{Float64}`: The optimal portfolio weight vector.
"""
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


"""
    solve_dynamic_program(
        W_grid::Vector{Float64}, Z_grids::Vector{Vector{Float64}},
        c_grid::Vector{Float64}, omega_space::Vector{Vector{Float64}},
        ε_nodes::Vector{Vector{Float64}}, W_weights::Vector{Float64},
        transition_model::Function, M::Int, β::Float64, u::Function,
        compute_consumption::Function, budget_constraint::Function, extrapolator::Function
    )

Solves a finite-horizon dynamic programming problem for portfolio and consumption choice
using backwards recursion.

The solver evaluates the problem over `M` time steps. It builds multidimensional linear
interpolations of the future value function at each step and executes the state-space evaluation
in parallel across available threads. The solver is purely agnostic and relies entirely on the
injected strategy functions to dictate the economic structure of the model.

# Arguments
- `W_grid::Vector{Float64}`: The grid for the principal state variable (e.g., Wealth or Log-Wealth).
- `Z_grids::Vector{Vector{Float64}}`: A list of grids for auxiliary state variables.
- `c_grid::Vector{Float64}`: The grid of valid consumption choices.
- `omega_space::Vector{Vector{Float64}}`: The space of valid portfolio weight vectors.
- `ε_nodes::Vector{Vector{Float64}}`: Multidimensional quadrature nodes for the expectation integral.
- `W_weights::Vector{Float64}`: Corresponding multidimensional quadrature weights.
- `transition_model::Function`: The market dynamics generator.
- `M::Int`: The total number of decision time steps (excluding the terminal date).
- `β::Float64`: The subjective discount factor.
- `u::Function`: The pure utility function.
- `compute_consumption::Function`: The strategy defining how controls translate into physical consumption.
- `budget_constraint::Function`: The strategy defining how the principal state variable evolves over time.
- `extrapolator::Function`: The strategy defining how to handle evaluations outside the `W_grid` boundaries.

# Returns
A 3-tuple of multidimensional arrays `(V, pol_c, pol_w)`:
- `V`: The value function array of shape `(length(W_grid), length.(Z_grids)..., M + 1)`.
- `pol_c`: The optimal consumption policy array of shape `(length(W_grid), length.(Z_grids)..., M)`.
- `pol_w`: The optimal portfolio policy array of shape `(length(W_grid), length.(Z_grids)..., M)`.
"""
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