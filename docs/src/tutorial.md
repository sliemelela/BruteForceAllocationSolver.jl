# Tutorial: Getting Started
Welcome to the `BruteForceAllocationSolver.jl` tutorial!

This package solves discrete-time and continuous-time consumption-portfolio
choice problems using Bellman backwards recursion and Gauss-Hermite
quadrature.

A core feature of this package is its **"Agnostic Engine."** The solver does
not hardcode budget constraints, consumption rules, or boundary
extrapolations. Instead, it relies on the **Strategy Pattern**—you inject
the economic rules into the solver, allowing you to easily switch between
absolute wealth spaces, log-wealth spaces, and complex non-tradeable
income models.

---

## 1. Choosing a Solver Strategy
The package provides three distinct algorithms for finding the optimal
controls ($c$ and $\omega$) at each grid point. You select
these by passing a specific solver struct to the `solve_dynamic_program`
function.

| Solver | Method | Best For... |
| :--- | :--- | :--- |
| **`BruteForceSolver`** | Exhaustive Grid Search | Robustness; guaranteed
to find the best point on your defined grid. |
| **`ZoomingSolver`** | Iterative Refinement | Speed; performs a coarse
sweep and then "zooms in" on the peak. |
| **`OptimSolver`** | Continuous Optimization | Precision; uses `Optim.jl`
(L-BFGS) to find exact optima between grid points. |

### The `OptimSolver` and Agnostic Scaling
Continuous optimizers often struggle with economic utility functions because
the values can become microscopically small (e.g., $10^{-8}$), causing the
solver to think the surface is "flat" and stop early.

The `OptimSolver` in this package automatically applies **Agnostic Scaling**:
it evaluates your warm-start guess and divides the entire objective function
by that magnitude. This "stretches" the mountain so the optimizer can always
see the slope, regardless of whether you are using CRRA, CARA, or
Epstein-Zin utility.

---

## Example 1: The Terminal Wealth Merton Problem
In this classic problem, an investor allocates wealth between a risk-free
asset and a single risky asset. We maximize utility at terminal date $T$.


### 1. Setup Parameters and Grids
```julia
using BruteForceAllocationSolver
using StaticArrays

# Define Model Parameters
M = 10           # Number of timesteps
dt = 1.0         # Step size (years)
γ = 5.0          # Coefficient of relative risk aversion
u(W) = (W^(1 - γ)) / (1 - γ)

# Setup Grids
W_grid = generate_log_spaced_grid(1.0, 100.0, 200)
Z_grids = Vector{Float64}[] # No auxiliary state variables
omega_space = [SVector(ω) for ω in generate_linear_grid(0.0, 1.0, 101)]
```

### 2. Integration Nodes
We use Gauss-Hermite quadrature to evaluate the Bellman expectation.
```julia
ρ_mat = fill(1.0, 1, 1)
ε_nodes, W_weights = generate_gaussian_shocks(1, 10, ρ_mat)
```

### 3. Injecting Strategies and Solving
```julia
# Define Market Dynamics
r, μ, σ = 0.02, 0.07, 0.20
merton_transition = make_merton_transition(r, μ, σ, dt)
crra_extrapolator = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

# Run the Solver using BruteForce
V, pol_w = solve_dynamic_program(
    BruteForceSolver(),
    W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, u, identity, standard_budget_constraint, crra_extrapolator
)
```

---

## Example 2: Using the Continuous `OptimSolver`
If you need higher precision than a fixed grid allows, use the `OptimSolver`.

```julia
using Optim

# Define transformation functions for Parameter Mapping
# Initialize the continuous solver
continuous_solver = OptimSolver(
    method = LBFGS(),
    use_gradients = true
)

# Solve with the exact same API
V_opt, pol_w_opt = solve_dynamic_program(
    continuous_solver,
    W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, u, identity, standard_budget_constraint, crra_extrapolator
)
```

---

## Example 3: Working in Log-Wealth Space
Formulating states as $X = \log(W)$ captures high curvature near zero more
efficiently. Swap the strategies to handle log-math.

```julia
X_grid = generate_linear_grid(log(0.01), log(100.0), 200)
log_extrapolator = make_log_crra_extrapolator(X_grid[1], X_grid[end], γ)

V_log, pol_w_log = solve_dynamic_program(
    BruteForceSolver(),
    X_grid, Z_grids, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, u,
    exp,                          # Strategy: un-log state for utility u(W)
    log_budget_constraint,        # Strategy: Additive log-wealth evolution
    log_extrapolator
)
```

---

## Example 4: Intermediate Consumption
For agents drawing utility from consumption over time, introduce `β` and a
`c_grid`.

```julia
β = 0.96
c_grid = generate_linear_grid(0.01, 0.99, 50)

V_cons, pol_c, pol_w_cons = solve_dynamic_program(
    BruteForceSolver(),
    W_grid, Z_grids, c_grid, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, β, u,
    fractional_consumption,       # Strategy: c is fraction of W
    standard_budget_constraint,
    crra_extrapolator
)
```

---

## Example 5: Certainty Equivalent (CE)
CE translates abstract utility "utils" back into dollar amounts.

```julia
inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# For Terminal Wealth: guaranteed terminal amount
ce_wealth = calculate_certainty_equivalent(V[100, 1], inv_u)

# For Consumption: guaranteed constant period-by-period stream
periods = M - 1 + 1
cec_stream = calculate_equivalent_consumption_stream(V_cons[100, 1],
                                                     inv_u, β, periods)
```