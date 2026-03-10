# Tutorial: Getting Started
Welcome to the `BruteForceAllocationSolver.jl` tutorial!

This package solves discrete-time and continuous-time consumption-portfolio choice problems
using Bellman backwards recursion and Gauss-Hermite quadrature.

A core feature of this package is its **"Agnostic Engine."**
The solver does not hardcode budget constraints, consumption rules, or boundary extrapolations.
Instead, it relies on the **Strategy Pattern**—you inject the economic rules into the solver,
allowing you to easily switch between absolute wealth spaces, log-wealth spaces, and complex
non-tradeable income models.

Let's walk through two examples to see this in action.

## Example 1: The Standard Merton Problem
In this classic problem, an investor allocates their wealth between a risk-free asset and a single
risky asset following Geometric Brownian Motion, while deciding how much of their wealth to consume ]
at each timestep.

### 1. Setup Parameters and Grids
First, we define our economic parameters and generate the grids for our state and control variables.
We can use the package's built-in grid generators for convenience.
```julia
using BruteForceAllocationSolver
using FastGaussQuadrature

# Define Model Parameters
M = 10           # Number of timesteps
dt = 1.0         # Step size (years)
β = 0.96         # Discount factor
γ = 5.0          # Coefficient of relative risk aversion

# Pure CRRA Utility (evaluates absolute consumption)
u(C) = (C^(1 - γ)) / (1 - γ)

# Setup Grids
G_w = 200        # Number of wealth grid points
W_min = 1.0
W_max = 100.0

# Generate an exponentially spaced grid to capture high curvature near zero
W_grid = generate_log_spaced_grid(W_min, W_max, G_w)
Z_grids = Vector{Float64}[] # No auxiliary state variables in this model

# Control grids: Consumption fraction (0 to 1) and Portfolio weight (0 to 1)
c_grid = generate_linear_grid(0.01, 0.99, 50)
omega_space = [[w] for w in generate_linear_grid(0.0, 1.0, 101)]
```

### 2. Integration Nodes
Because the financial returns are driven by normally distributed shocks,
we use Gauss-Hermite quadrature to evaluate the Bellman expectation accurately.

```julia
# Setup Quadrature Nodes
Q = 10
nodes, weights = gausshermite(Q)

# Transform to standard normal distribution space
ε_nodes = [[n * sqrt(2.0)] for n in nodes]
W_weights = weights ./ sqrt(pi)
```

### 3. Injecting the Strategies and Solving
Now we inject the economic rules. We need a market transition model,
a boundary extrapolator (to prevent interpolation errors if wealth falls off the grid), a budget constraint, and a consumption rule.
```julia
# Define Market Dynamics and Strategies
r = 0.02
μ = 0.07
σ = 0.20

# Create the closures using the built-in factories
merton_transition = make_merton_transition(r, μ, σ, dt)
crra_extrapolator = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

# Run the Solver!
V, pol_c, pol_w = solve_dynamic_program(
    W_grid, Z_grids, c_grid, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, β, u,
    fractional_consumption,       # Strategy: state is W, control is fraction
    standard_budget_constraint,   # Strategy: multiplicative wealth evolution
    crra_extrapolator             # Strategy: CRRA asymptotic boundaries
)

# Validate against the analytical continuous-time solution
analytical_w = (μ - r) / (γ * σ^2) # ≈ 0.25
println("Numerical Portfolio Share at W=50: ", round(pol_w[100, 1][1], digits=4))
println("Analytical Merton Share: ", round(analytical_w, digits=4))
```


## Example 2: Working in Log-Wealth Space
Because the value function is highly curved, working in absolute wealth levels can require a
massive grid to maintain accuracy. Alternatively, you can formulate your state variable as $X = \log(W)$.

Because of the **Strategy Pattern** architecture,
we don't have to rewrite the solver to do this! We simply swap out the injected rules to
handle log-space mathematics.
```julia
# Define a standard linear grid for Log-Wealth (X)
# log(0.01) ≈ -4.6, log(100.0) ≈ 4.6
G_X = 200
X_grid = generate_linear_grid(log(0.01), log(100.0), G_X)

# Setup the Log-Space Strategies
# Notice how we use the log-specific factories provided by the package!
log_extrapolator = make_log_crra_extrapolator(X_grid[1], X_grid[end], γ)

# Run the solver with the new injected strategies
V_log, pol_c_log, pol_w_log = solve_dynamic_program(
    X_grid, Z_grids, c_grid, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, β, u,
    log_fractional_consumption,   # Strategy: Translates log-wealth & fraction to absolute consumption
    log_budget_constraint,        # Strategy: Additive log-wealth evolution
    log_extrapolator              # Strategy: Stitches exponential CRRA curves on the log-boundaries
)

println("Log-Space Numerical Portfolio Share: ", round(pol_w_log[100, 1][1], digits=4))
```