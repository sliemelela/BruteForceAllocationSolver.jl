using FastGaussQuadrature
using Interpolations

# ==========================================
# 1. Model Parameters (Test Case 1)
# ==========================================
const γ = 10.0           # Coefficient of relative risk aversion
const β = 0.96          # Subjective discount factor
const M = 10            # Number of time steps
const dt = 1.0          # Step size (delta t)

# Asset Parameters
const r = 0.02          # Risk-free rate
const μ = 0.07          # Risky asset expected return
const σ = 0.20          # Risky asset volatility

# ==========================================
# 2. Grid & Quadrature Setup
# ==========================================
# Wealth Grid: clustered slightly more toward the lower end (log-spaced)
const G_w = 100
const W_min = 0.1
const W_max = 50.0
const W_grid = exp.(range(log(W_min), log(W_max), length=G_w))

# Control Grids (Brute force search as requested)
const c_grid = range(0.00, 1.0, length=20)
# const c_grid = [0.0]
const w_grid = range(0.0, 1.5, length=101)

# Gauss-Hermite Quadrature (Q nodes)
const Q = 10
nodes, weights = gausshermite(Q)
# Transform to standard normal distribution
const z_nodes = nodes .* sqrt(2.0)
const z_weights = weights ./ sqrt(pi)

# ==========================================
# 3. Core Functions
# ==========================================
# CRRA Utility
function u(W, γ)
    return (W^(1 - γ)) / (1 - γ)
end

# Realized Returns based on Geometric Brownian Motion discretization
function get_returns(eps, r, μ, σ, dt)
    Rf = exp(r * dt)
    R_risky = exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * eps)
    Re = R_risky - Rf
    return Rf, Re
end

# ==========================================
# 4. Backwards Recursion Algorithm
# ==========================================
function solve_merton_problem()
    # Pre-allocate storage
    V = zeros(G_w, M + 1)
    pol_c = zeros(G_w, M)
    pol_w = zeros(G_w, M)

    # Terminal Condition (n = M + 1)
    # Assume consumption of all wealth at terminal date
    V[:, M+1] .= [u(W, γ) for W in W_grid]

    println("Starting backwards recursion...")

    for n in M:-1:1
        # Create an interpolation object for the next period's value function
        # We use linear interpolation with linear extrapolation for out-of-bounds
        V_next_interp = linear_interpolation(W_grid, V[:, n+1], extrapolation_bc=Line())

        for (i, W) in enumerate(W_grid)
            best_val = -Inf
            best_c = 0.0
            best_w = 0.0

            # Control Loop (Brute Force Grid Search)
            for c in c_grid
                for ω in w_grid

                    expected_future_value = 0.0

                    # Quadrature Integration
                    for j in 1:Q
                        eps = z_nodes[j]
                        weight = z_weights[j]

                        Rf, Re = get_returns(eps, r, μ, σ, dt)

                        # Realized next-period wealth
                        W_next = (1 - c) * W * (ω * Re + Rf)
                        # W_next = W * (ω * Re + Rf)

                        # Prevent negative wealth crashes in extreme extrapolation
                        W_next = max(W_next, 1e-5)

                        expected_future_value += weight * V_next_interp(W_next)
                    end

                    # Bellman Maximization
                    current_val = u(c * W, γ) + β * expected_future_value

                    if current_val > best_val
                        best_val = current_val
                        best_c = c
                        best_w = ω
                    end
                end
            end

            # Policy Storage
            V[i, n] = best_val
            pol_c[i, n] = best_c
            pol_w[i, n] = best_w
        end
    end

    println("Recursion complete.")
    return V, pol_c, pol_w
end

# Run the solver
V, pol_c, pol_w = solve_merton_problem()

# Analytical Check
analytical_w = (μ - r) / (γ * σ^2)
numerical_w = pol_w[end÷2, 1] # Check middle of the grid at t=1

println("\n--- Validation Check ---")
println("Analytical Merton Share: ", round(analytical_w, digits=4))
println("Numerical Portfolio Share: ", round(numerical_w, digits=4))