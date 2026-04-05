using BruteForceAllocationSolver
using StaticArrays
using Profile

# 1. Setup a medium-sized problem (big enough to take a few seconds, small enough not to wait forever)
M, dt, γ = 5, 1.0, 5.0
u(W) = (W^(1 - γ)) / (1 - γ)

G_w = 200
W_grid = generate_log_spaced_grid(1.0, 100.0, G_w)
Z_grids = [generate_linear_grid(0.0, 0.10, 5)] # 1 Auxiliary state

omega_space = [SVector(w) for w in range(0.0, 1.0, length=201)]
ε_nodes, W_weights = generate_gaussian_shocks(2, 5, [1.0 0.0; 0.0 1.0])

transition = make_stochastic_r_constant_mu_transition(0.1, 0.03, 0.01, 0.07, 0.20, 0.0, dt)
extrap = make_log_crra_extrapolator(W_grid[1], W_grid[end], γ)

# 2. WARM-UP RUN (Crucial!)
# We must run it once first to compile the code. If you profile the compilation phase, the results are useless.
println("Warming up (compiling)...")
solve_dynamic_program(
    BruteForceSolver(), W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition, M, u, exp, log_budget_constraint, extrap
)

# 3. PROFILING RUN
println("Profiling...")
@profview solve_dynamic_program(
    BruteForceSolver(), W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition, M, u, exp, log_budget_constraint, extrap
)

# Alternatively, if you want to see exact memory allocations printed to the REPL, use @time:
@benchmark solve_dynamic_program(
    BruteForceSolver(), W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition, M, u, exp, log_budget_constraint, extrap
)
