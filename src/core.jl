"""
    evaluate_bellman_objective(
        W_n::Float64, c_n::Float64, ω_n,
        shocks_precalc, W_weights::Vector{Float64},
        V_next::VType, β::Float64, u::UFunc,
        compute_consumption::CCFunc, budget_constraint::BFunc,
        extrapolator::EFunc
    )

Evaluates the objective function using strictly typed precalculated transitions.
This function is called inside the control grid search and is allocation-free.
"""
function evaluate_bellman_objective(
    W_n::Real, c_n::Real, ω_n,
    shocks_precalc::AbstractVector, W_weights::Vector{Float64},
    V_next::VType, β::Float64, u::UFunc,
    compute_consumption::CCFunc, budget_constraint::BFunc, extrapolator::EFunc
) where {VType, UFunc, CCFunc, BFunc, EFunc}

    expected_future_value = 0.0

    @inbounds for j in 1:length(W_weights)
        weight = W_weights[j]
        Z_next, R_e, R_base = shocks_precalc[j]
        W_next = budget_constraint(W_n, c_n, ω_n, R_e, R_base)
        expected_future_value += weight * extrapolator(W_next, Z_next, V_next)
    end

    absolute_consumed = compute_consumption(W_n, c_n)
    current_utility = c_n > 0.0 ? u(absolute_consumed) : -Inf

    return current_utility + β * expected_future_value
end

function evaluate_bellman_objective(
    W_n::Real, ω_n,
    shocks_precalc::AbstractVector, W_weights::Vector{Float64},
    V_next::VType, budget_constraint::BFunc, extrapolator::EFunc
) where {VType, BFunc, EFunc}

    expected_future_value = 0.0

    @inbounds for j in 1:length(W_weights)
        weight = W_weights[j]
        Z_next, R_e, R_base = shocks_precalc[j]
        W_next = budget_constraint(W_n, 0.0, ω_n, R_e, R_base)
        expected_future_value += weight * extrapolator(W_next, Z_next, V_next)
    end

    return expected_future_value
end

"""
    optimize_controls(solver::BruteForceSolver, ...)

Finds the optimal controls using an exhaustive search over the entire grid.
"""
function optimize_controls(
    solver::BruteForceSolver,
    W_n::Float64, Z_n, c_grid::Vector{Float64}, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    V_next::VType, transition_model::TMod, β::Float64, u::UFunc,
    compute_consumption::CCFunc, budget_constraint::BFunc,
    extrapolator::EFunc
) where {VType, TMod, UFunc, CCFunc, BFunc, EFunc}

    shocks_precalc = [transition_model(Z_n, ε) for ε in ε_nodes]

    best_val = -Inf
    best_c = 0.0
    best_ω = first(omega_space)

    for c_n in c_grid, ω_n in omega_space
        current_val = evaluate_bellman_objective(
            W_n, c_n, ω_n, shocks_precalc, W_weights,
            V_next, β, u, compute_consumption, budget_constraint, extrapolator
        )

        if current_val > best_val
            best_val = current_val
            best_c = c_n
            best_ω = ω_n
        end
    end

    return best_val, best_c, best_ω
end

function optimize_controls(
    solver::BruteForceSolver,
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
            W_n, ω_n, shocks_precalc, W_weights,
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
    optimize_controls(solver::ZoomingSolver, ...)

Finds the optimal controls by first performing a coarse global sweep, then
iteratively constructing a tighter sub-grid around the current best candidate.
"""
function optimize_controls(
    solver::ZoomingSolver,
    W_n::Float64, Z_n, c_grid::Vector{Float64}, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    V_next::VType, transition_model::TMod, β::Float64, u::UFunc,
    compute_consumption::CCFunc, budget_constraint::BFunc,
    extrapolator::EFunc
) where {VType, TMod, UFunc, CCFunc, BFunc, EFunc}

    best_val, best_c, best_ω = optimize_controls(
        BruteForceSolver(), W_n, Z_n, c_grid, omega_space, ε_nodes, W_weights,
        V_next, transition_model, β, u,
        compute_consumption, budget_constraint, extrapolator
    )

    shocks_precalc = [transition_model(Z_n, ε) for ε in ε_nodes]

    c_min, c_max = minimum(c_grid), maximum(c_grid)
    ω_min = minimum(reduce(hcat, omega_space), dims=2)
    ω_max = maximum(reduce(hcat, omega_space), dims=2)

    cur_c, cur_ω = best_c, best_ω

    for _ in 1:solver.iterations
        step_c = (c_max - c_min) / length(c_grid) * solver.zoom_range_factor
        local_c_grid = range(max(c_min, cur_c - step_c),
                             min(c_max, cur_c + step_c),
                             length=solver.points_per_dim)

        local_ω_ranges = []
        for d in 1:length(cur_ω)
            step_ω = (ω_max[d] - ω_min[d]) / solver.points_per_dim * solver.zoom_range_factor

            push!(local_ω_ranges, range(max(ω_min[d], cur_ω[d] - step_ω),
                                        min(ω_max[d], cur_ω[d] + step_ω),
                                        length=solver.points_per_dim))
        end

        for c_test in local_c_grid
            for ω_vals in Iterators.product(local_ω_ranges...)
                ω_test = SVector(ω_vals)
                val = evaluate_bellman_objective(
                    W_n, c_test, ω_test, shocks_precalc, W_weights,
                    V_next, β, u, compute_consumption,
                    budget_constraint, extrapolator
                )
                if val > best_val
                    best_val, cur_c, cur_ω = val, c_test, ω_test
                end
            end
        end
    end

    return best_val, cur_c, cur_ω
end

function optimize_controls(
    solver::ZoomingSolver,
    W_n::Float64, Z_n, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    V_next::VType, transition_model::TMod,
    budget_constraint::BFunc, extrapolator::EFunc
) where {VType, TMod, BFunc, EFunc}

    best_val, best_ω = optimize_controls(
        BruteForceSolver(), W_n, Z_n, omega_space, ε_nodes, W_weights,
        V_next, transition_model, budget_constraint, extrapolator
    )

    shocks_precalc = [transition_model(Z_n, ε) for ε in ε_nodes]

    ω_min = minimum(reduce(hcat, omega_space), dims=2)
    ω_max = maximum(reduce(hcat, omega_space), dims=2)

    cur_ω = best_ω

    for _ in 1:solver.iterations
        local_ω_ranges = []
        for d in 1:length(cur_ω)
            step_ω = (ω_max[d] - ω_min[d]) / solver.points_per_dim * solver.zoom_range_factor

            push!(local_ω_ranges, range(max(ω_min[d], cur_ω[d] - step_ω),
                                        min(ω_max[d], cur_ω[d] + step_ω),
                                        length=solver.points_per_dim))
        end

        for ω_vals in Iterators.product(local_ω_ranges...)
            ω_test = SVector(ω_vals)
            val = evaluate_bellman_objective(
                W_n, ω_test, shocks_precalc, W_weights,
                V_next, budget_constraint, extrapolator
            )
            if val > best_val
                best_val, cur_ω = val, ω_test
            end
        end
    end

    return best_val, cur_ω
end

"""
    optimize_controls(solver::OptimSolver, ...)

Finds the optimal controls using continuous optimization methods via Optim.jl.
Utilizes Fminbox to ensure boundary compliance.
"""
function optimize_controls(
    solver::OptimSolver,
    W_n::Float64, Z_n, c_grid::Vector{Float64}, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    V_next::VType, transition_model::TMod, β::Float64, u::UFunc,
    compute_consumption::CCFunc, budget_constraint::BFunc,
    extrapolator::EFunc;
    warm_start_guess=nothing
) where {VType, TMod, UFunc, CCFunc, BFunc, EFunc}

    shocks_precalc = [transition_model(Z_n, ε) for ε in ε_nodes]

    function raw_obj(x)
        c_val = x[1]
        ω_val = SVector{length(omega_space[1])}(x[2:end])
        return -evaluate_bellman_objective(
            W_n, c_val, ω_val, shocks_precalc, W_weights,
            V_next, β, u, compute_consumption, budget_constraint, extrapolator
        )
    end

    x0 = if warm_start_guess !== nothing
        warm_start_guess
    else
        coarse_c = collect(range(minimum(c_grid), maximum(c_grid),
                           length=solver.coarse_warm_start_n))

        step_ω = max(1, div(length(omega_space), solver.coarse_warm_start_n))
        coarse_ω = [omega_space[i] for i in 1:step_ω:length(omega_space)]

        _, bc, bω = optimize_controls(
            BruteForceSolver(), W_n, Z_n, coarse_c, coarse_ω,
            ε_nodes, W_weights, V_next, transition_model, β, u,
            compute_consumption, budget_constraint, extrapolator
        )
        vcat(bc, bω...)
    end

    base_val = abs(raw_obj(x0))
    scale_factor = base_val > 1e-10 ? base_val : 1.0
    obj(x) = raw_obj(x) / scale_factor

    lower = vcat(minimum(c_grid), [minimum(reduce(hcat, omega_space), dims=2)...])
    upper = vcat(maximum(c_grid), [maximum(reduce(hcat, omega_space), dims=2)...])

    x0 = clamp.(x0, lower .+ 1e-5, upper .- 1e-5)

    res = if solver.use_gradients
        g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
        od = OnceDifferentiable(obj, g!, x0)
        optimize(od, lower, upper, x0, Fminbox(solver.method), solver.optim_options)
    else
        optimize(obj, lower, upper, x0, Fminbox(solver.method), solver.optim_options)
    end

    sol = Optim.minimizer(res)
    return -Optim.minimum(res) * scale_factor, sol[1], SVector{length(omega_space[1])}(sol[2:end])
end

function optimize_controls(
    solver::OptimSolver,
    W_n::Float64, Z_n, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    V_next::VType, transition_model::TMod,
    budget_constraint::BFunc, extrapolator::EFunc;
    warm_start_guess=nothing
) where {VType, TMod, BFunc, EFunc}

    shocks_precalc = [transition_model(Z_n, ε) for ε in ε_nodes]

    function raw_obj(x)
        ω_val = SVector{length(omega_space[1])}(x)
        return -evaluate_bellman_objective(
            W_n, ω_val, shocks_precalc, W_weights,
            V_next, budget_constraint, extrapolator
        )
    end

    x0 = if warm_start_guess !== nothing
        warm_start_guess
    else
        step_ω = max(1, div(length(omega_space), solver.coarse_warm_start_n))
        coarse_ω = [omega_space[i] for i in 1:step_ω:length(omega_space)]

        _, bω = optimize_controls(
            BruteForceSolver(), W_n, Z_n, coarse_ω,
            ε_nodes, W_weights, V_next, transition_model,
            budget_constraint, extrapolator
        )
        Vector(bω)
    end

    base_val = abs(raw_obj(x0))
    scale_factor = base_val > 1e-10 ? base_val : 1.0
    obj(x) = raw_obj(x) / scale_factor

    lower = vec(minimum(reduce(hcat, omega_space), dims=2))
    upper = vec(maximum(reduce(hcat, omega_space), dims=2))

    x0 = clamp.(x0, lower .+ 1e-5, upper .- 1e-5)

    res = if solver.use_gradients
        g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
        od = OnceDifferentiable(obj, g!, x0)
        optimize(od, lower, upper, x0, Fminbox(solver.method), solver.optim_options)
    else
        optimize(obj, lower, upper, x0, Fminbox(solver.method), solver.optim_options)
    end

    sol = Optim.minimizer(res)
    return -Optim.minimum(res) * scale_factor, SVector{length(omega_space[1])}(sol)
end

"""
    solve_dynamic_program(...)

Solves a finite-horizon dynamic programming problem for portfolio and
consumption choice using backwards recursion. Dispatches to the provided
AbstractAllocationSolver.
"""
function solve_dynamic_program(
    solver::AbstractAllocationSolver,
    W_grid::Vector{Float64}, Z_grids::Vector{Vector{Float64}},
    c_grid::Vector{Float64}, omega_space,
    ε_nodes, W_weights::Vector{Float64},
    transition_model::Function, M::Int, β::Float64, u::Function,
    compute_consumption::Function, budget_constraint::Function,
    extrapolator::Function
)
    sz = (length(W_grid), (length(z) for z in Z_grids)...)
    V     = zeros(Float64, sz..., M + 1)
    pol_c = zeros(Float64, sz..., M)
    pol_w = Array{eltype(omega_space)}(undef, sz..., M)

    for idx in CartesianIndices(sz)
        state_terminal = W_grid[idx[1]]
        C_terminal = compute_consumption(state_terminal, 1.0)
        V[idx, M+1] = u(C_terminal)
    end

    for n in M:-1:1
        V_next_data = selectdim(V, ndims(V), n + 1)
        V_next_interp = linear_interpolation(
            (W_grid, Z_grids...), V_next_data, extrapolation_bc=Line()
        )

        Threads.@threads for idx in CartesianIndices(sz)
            W_n = W_grid[idx[1]]
            Z_n = ntuple(k -> Z_grids[k][idx[1+k]], length(Z_grids))

            best_val, best_c, best_ω = optimize_controls(
                solver, W_n, SVector(Z_n), c_grid, omega_space,
                ε_nodes, W_weights, V_next_interp, transition_model, β, u,
                compute_consumption, budget_constraint, extrapolator
            )

            V[idx, n]     = best_val
            pol_c[idx, n] = best_c
            pol_w[idx, n] = best_ω
        end
    end

    return V, pol_c, pol_w
end

function solve_dynamic_program(
    solver::AbstractAllocationSolver,
    W_grid::Vector{Float64}, Z_grids::Vector{Vector{Float64}},
    omega_space,
    ε_nodes, W_weights::Vector{Float64},
    transition_model::Function, M::Int, u::Function,
    state_to_wealth::Function, budget_constraint::Function,
    extrapolator::Function
)
    sz = (length(W_grid), (length(z) for z in Z_grids)...)
    V     = zeros(Float64, sz..., M + 1)
    pol_w = Array{eltype(omega_space)}(undef, sz..., M)

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

            best_val, best_ω = optimize_controls(
                solver, W_n, SVector(Z_n), omega_space,
                ε_nodes, W_weights, V_next_interp, transition_model,
                budget_constraint, extrapolator
            )

            V[idx, n]     = best_val
            pol_w[idx, n] = best_ω
        end
    end

    return V, pol_w
end