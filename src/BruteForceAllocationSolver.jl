module BruteForceAllocationSolver

using LinearAlgebra
using ForwardDiff
using Interpolations
using Integrals
using FastGaussQuadrature
using Statistics
using CairoMakie

# Export the core solver
export solve_dynamic_program

# Export the built-in strategies
export fractional_consumption, log_fractional_consumption, absolute_consumption
export make_crra_extrapolator, standard_budget_constraint
export make_log_crra_extrapolator, log_budget_constraint

# Export the transition models
export make_merton_transition
export make_stochastic_r_constant_mu_transition, make_stochastic_r_constant_premium_transition

# Export grid tools
export generate_linear_grid, generate_log_spaced_grid, generate_adaptive_grid

# Export the analysis and plotting tools
export create_policy_interpolators
export plot_mean_with_bounds, plot_shock_comparison, plot_policy_vs_state

# Include the separated files
include("core.jl")
include("strategies.jl")
include("transitions.jl")
include("grids.jl")
include("analysis.jl")


end