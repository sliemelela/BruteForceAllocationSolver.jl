# Tutorial: Getting Started
Welcome to the `BruteForceAllocationSolver.jl` tutorial!

This package solves discrete-time and continuous-time consumption-portfolio
choice problems using Bellman backwards recursion and Gauss-Hermite
quadrature.

A core feature of this package is its **"Agnostic Engine."** The solver
does not hardcode budget constraints, consumption rules, or boundary
extrapolations. Instead, it relies on the **Strategy Pattern**—you inject
the economic rules into the solver, allowing you to easily switch between
absolute wealth spaces, log-wealth spaces, and complex non-tradeable
income models.

Thanks to Julia's multiple dispatch, the solver can elegantly handle both
problems with **intermediate consumption**, as well as **pure Terminal Wealth problems**,
seamlessly adapting its performance and API based on the
arguments you provide.

Let's walk through two Terminal Wealth examples to see this in action.

## Example 1: The Terminal Wealth Merton Problem
In this classic problem, an investor allocates their wealth between a
risk-free asset and a single risky asset following Geometric Brownian
Motion. Because we are only interested in maximizing the utility of wealth
at the terminal date $T$, there is no intermediate consumption and no
subjective discount factor $\beta$.

### 1. Setup Parameters and Grids
First, we define our economic parameters and generate the grids for our
state and control variables.

```julia
using BruteForceAllocationSolver

# Define Model Parameters
M = 10           # Number of timesteps
dt = 1.0         # Step size (years)
γ = 5.0          # Coefficient of relative risk aversion

# Pure CRRA Utility evaluated on terminal physical wealth
u(W) = (W^(1 - γ)) / (1 - γ)

# Setup Grids
G_w = 200        # Number of wealth grid points
W_min = 1.0
W_max = 100.0

# Generate an exponentially spaced grid to capture high curvature near zero
W_grid = generate_log_spaced_grid(W_min, W_max, G_w)
Z_grids = Vector{Float64}[] # No auxiliary state variables in this model

# Control grid: Portfolio weight. Notice we do NOT define a consumption grid!
omega_space = [[w] for w in generate_linear_grid(0.0, 1.0, 101)]
```

### 2. Integration Nodes
Because the financial returns are driven by normally distributed shocks,
we use Gauss-Hermite quadrature to evaluate the Bellman expectation
accurately. The package provides a convenient built-in generator for
multidimensional correlated shocks.

```julia
# Setup 1D Quadrature Nodes for the single risky asset
ρ_mat = fill(1.0, 1, 1) # 1x1 correlation matrix
ε_nodes, W_weights = generate_gaussian_shocks(1, 10, ρ_mat)
```

### 3. Injecting the Strategies and Solving
Now we inject the economic rules. Because we are solving a Terminal Wealth
problem, we omit the `c_grid`, `compute_consumption`, and `β` arguments.

Crucially, we pass `identity` as the `state_to_wealth` strategy. This
simply tells the solver that the values in our `W_grid` already represent
absolute physical wealth before they are passed into the utility function.

```julia
# Define Market Dynamics and Strategies
r = 0.02
μ = 0.07
σ = 0.20

# Create the closures using the built-in factories
merton_transition = make_merton_transition(r, μ, σ, dt)
crra_extrapolator = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

# Run the Solver!
V, pol_w = solve_dynamic_program(
    W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, u,
    identity,                     # Strategy: W_grid is physical wealth
    standard_budget_constraint,   # Strategy: multiplicative wealth evolution
    crra_extrapolator             # Strategy: CRRA asymptotic boundaries
)

# Validate against the analytical continuous-time solution
analytical_w = (μ - r) / (γ * σ^2) # ≈ 0.25
println("Numerical Portfolio Share: ", round(pol_w[100, 1][1], digits=4))
println("Analytical Merton Share: ", round(analytical_w, digits=4))
```


## Example 2: Working in Log-Wealth Space
Because the value function is highly curved, working in absolute wealth
levels can require a massive grid to maintain accuracy. Alternatively, you
can formulate your state variable as $X = \log(W)$.

Because of the **Strategy Pattern** architecture, we don't have to rewrite
the Bellman recursion engine to do this. We simply swap out the injected
rules to handle log-space mathematics.

```julia
# Define a standard linear grid for Log-Wealth (X)
# log(0.01) ≈ -4.6, log(100.0) ≈ 4.6
G_X = 200
X_grid = generate_linear_grid(log(0.01), log(100.0), G_X)

# Setup the Log-Space Boundary Strategy
log_extrapolator = make_log_crra_extrapolator(X_grid[1], X_grid[end], γ)

# Run the solver with the new injected strategies!
# Notice we pass the built-in `exp` function to tell the solver how to
# un-log the state variable before passing it into the utility function u(W).
V_log, pol_w_log = solve_dynamic_program(
    X_grid, Z_grids, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, u,
    exp,                          # Strategy: X_grid is log-wealth, un-log it
    log_budget_constraint,        # Strategy: Additive log-wealth evolution
    log_extrapolator              # Strategy: Stitches CRRA curves to bounds
)

println("Log-Space Numerical Share: ", round(pol_w_log[100, 1][1], digits=4))
```

## Example 3: Adding Auxiliary State Variables ($Z$)
Real-world problems rarely have a constant investment opportunity set.
Variables like interest rates, dividend yields, or inflation often fluctuate
over time. The `BruteForceAllocationSolver` handles an arbitrary number of
auxiliary state variables natively.

To add state variables, you simply populate the `Z_grids` array and update
your integration nodes to handle the higher-dimensional shocks.

```julia
# 1. Define the auxiliary state grid (e.g., a stochastic interest rate 'r')
G_r = 11
r_grid = generate_linear_grid(0.0, 0.10, G_r)

# Add as many state grids as your transition model requires!
Z_grids = [r_grid]

# 2. Setup 2D Quadrature (1 shock for the stock, 1 for the interest rate)
ρ_val = 0.0 # Correlation between stock and rate shocks
ρ_mat = [1.0 ρ_val;
         ρ_val 1.0]

# Automatically generates 10^2 = 100 correlated multidimensional nodes
ε_nodes_2d, W_weights_2d = generate_gaussian_shocks(2, 10, ρ_mat)

# 3. Inject a transition model that uses Z
# Parameters: κ=0.1, θ=0.03, σ_r=0.01, λ_S=0.25, σ_S=0.20, ρ=0.0, dt=1.0
stoch_transition = make_stochastic_r_constant_premium_transition(
    0.1, 0.03, 0.01, 0.25, 0.20, 0.0, 1.0
)

# 4. Run the solver!
V_stoch, pol_w_stoch = solve_dynamic_program(
    X_grid, Z_grids, omega_space,
    ε_nodes_2d, W_weights_2d, stoch_transition,
    M, u,
    exp,                          # Strategy: X_grid is log-wealth
    log_budget_constraint,        # Strategy: Additive evolution
    log_extrapolator              # Strategy: CRRA boundaries
)

# The policy arrays dynamically expand to match the dimensionality of Z!
# pol_w_stoch[Wealth_idx, Rate_idx, Time_idx]
w_at_5_percent = pol_w_stoch[100, 6, 1][1]
println("Optimal Risky Weight at r=5%: ", round(w_at_5_percent, digits=4))
```


## Example 4: Solving with Intermediate Consumption
If the economic agent draws utility from consuming their wealth over time
rather than just hoarding it for a terminal date, the Bellman equation
requires an inner maximization loop over consumption choices.

Thanks to multiple dispatch, you do not need to call a different solver.
You simply introduce the subjective discount factor `β`, provide a `c_grid`,
and swap the `state_to_wealth` strategy for a `compute_consumption` strategy.
The package will automatically route your model to the full recursive engine.

```julia
# 1. Define the subjective discount factor
β = 0.96

# 2. Define the Consumption Control Grid (e.g., fraction of wealth)
G_c = 50
c_grid = generate_linear_grid(0.01, 0.99, G_c)

# 3. Run the solver with the FULL signature
# Notice we pass `c_grid`, `β`, and a `compute_consumption` strategy.
# We drop `state_to_wealth` because utility is evaluated on consumption!
V_cons, pol_c, pol_w_cons = solve_dynamic_program(
    W_grid, Z_grids, c_grid, omega_space,
    ε_nodes_2d, W_weights_2d, stoch_transition,
    M, β, u,
    fractional_consumption,       # Strategy: Convert state & 'c' to C
    standard_budget_constraint,   # Strategy: Multiplicative evolution
    crra_extrapolator             # Strategy: CRRA asymptotic boundaries
)

# The solver now returns THREE multidimensional arrays.
# pol_c contains the optimal consumption choice at every grid point.
optimal_c = pol_c[100, 6, 1]
println("Optimal Consumption Fraction: ", round(optimal_c, digits=4))
```

## Example 5: Calculating the Certainty Equivalent
The Value Function arrays (`V`, `V_log`, `V_cons`) returned by the solver contain abstract
"utils," which can be mathematically difficult to interpret.

To translate these abstract numbers back into understandable "dollar amounts,"
the package provides tools to calculate the Certainty Equivalent (CE).
This is the guaranteed, risk-free amount of wealth (or constant consumption stream)
that provides the exact same utility as taking the optimal risky path.

To compute it, you simply need to define the mathematical inverse of your chosen
utility function and pass it into the CE functions.

```julia
# 1. Define the mathematical inverse of the CRRA utility function
# u(x) = (x^(1-γ)) / (1-γ)
inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# --- Terminal Wealth Problems ---
# For problems without intermediate consumption (like Example 1),
# use `calculate_certainty_equivalent` to find the guaranteed terminal wealth.
target_timestep = 1
wealth_index = 100 # Checking the middle of our W_grid

abstract_utility = V[wealth_index, target_timestep]
guaranteed_wealth = calculate_certainty_equivalent(abstract_utility, inv_u)

println("Expected Utility at t=1: ", round(abstract_utility, digits=4))
println("Certainty Equivalent Terminal Wealth: ", round(guaranteed_wealth, digits=2))


# --- Intermediate Consumption Problems ---
# For problems with consumption streams (like Example 4), we use
# `calculate_equivalent_consumption_stream` to account for the discount factor over time.
# This returns the constant amount you would need to consume *every single period* # to achieve the same total lifetime utility.

periods_remaining = M - target_timestep + 1
abstract_lifetime_utility = V_cons[wealth_index, 6, target_timestep] # 6 is the middle rate index

constant_consumption = calculate_equivalent_consumption_stream(
    abstract_lifetime_utility,
    inv_u,
    β,
    periods_remaining
)

println("Expected Lifetime Utility at t=1: ", round(abstract_lifetime_utility, digits=4))
println("Equivalent Constant Consumption Stream: ", round(constant_consumption, digits=2))
```