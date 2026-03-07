module BruteForceAllocationSolver

using LinearAlgebra
using ForwardDiff
using Interpolations
using Integrals
using FastGaussQuadrature

# Export the core solver
export solve_dynamic_program

# Export the built-in strategies
export make_crra_extrapolator, standard_budget_constraint
export make_log_crra_extrapolator, log_budget_constraint

# Export grid tools
export generate_adaptive_grid

# Include the separated files
include("core.jl")
include("strategies.jl")
include("grids.jl")

end