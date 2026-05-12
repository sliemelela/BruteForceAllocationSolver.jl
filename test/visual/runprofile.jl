using BruteForceAllocationSolver
using StaticArrays
using BenchmarkTools
using Profile

# 1. Setup a medium-sized problem
M, dt = 5, 1.0

const γ = 5.0
const one_minus_γ = 1.0 - γ
const inv_power = 1.0 / one_minus_γ

u(W) = (W^one_minus_γ) / one_minus_γ
inv_u(v) = (one_minus_γ * v)^inv_power

G_w = 200
X_grid = generate_linear_grid(log(1.0), log(100.0), G_w)
Z_grids = [generate_linear_grid(0.0, 0.10, 5)]

omega_space = [SVector(w) for w in range(0.0, 1.0, length=201)]
ε_nodes, W_weights = generate_gaussian_shocks(2, 5, [1.0 0.0; 0.0 1.0])

transition = make_stochastic_r_constant_mu_transition(0.1, 0.03, 0.01, 0.07, 0.20, 0.0, dt)
extrap = make_ce_log_crra_extrapolator(X_grid[1], X_grid[end])

println("Warming up (compiling)...")
solve_dynamic_program(
    BruteForceSolver(), X_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition, M, u, inv_u, exp, log_budget_constraint, extrap
)

println("Profiling...")
@benchmark solve_dynamic_program(
    BruteForceSolver(), X_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition, M, u, inv_u, exp, log_budget_constraint, extrap
)