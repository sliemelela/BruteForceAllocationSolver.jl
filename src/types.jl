abstract type AbstractAllocationSolver end

"""
    BruteForceSolver()

The standard exhaustive grid search solver. Evaluates every combination of
consumption and portfolio weights to find the global optimum on the defined grid.
"""
Base.@kwdef struct BruteForceSolver <: AbstractAllocationSolver
    # We can add parameters here later if needed (e.g., multithreading flags)
end