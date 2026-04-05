abstract type AbstractAllocationSolver end

"""
    BruteForceSolver()

The standard exhaustive grid search solver. Evaluates every combination of
consumption and portfolio weights to find the global optimum on the defined grid.
"""
Base.@kwdef struct BruteForceSolver <: AbstractAllocationSolver
    # We can add parameters here later if needed (e.g., multithreading flags)
end

"""
    ZoomingSolver(iterations, points_per_dim, zoom_range_factor)

A hybrid solver that performs a coarse global sweep followed by iterative
refinements (zooming) around the local optimum.
"""
Base.@kwdef struct ZoomingSolver <: AbstractAllocationSolver
    iterations::Int = 3
    points_per_dim::Int = 10
    zoom_range_factor::Float64 = 1.5
end

"""
    OptimSolver(method, use_gradients, coarse_warm_start_n)

A continuous solver using Optim.jl. Supports Fminbox for constraints and
ForwardDiff for gradients.
"""
Base.@kwdef struct OptimSolver{T} <: AbstractAllocationSolver
    method::T = LBFGS()
    use_gradients::Bool = true
    coarse_warm_start_n::Int = 10
end