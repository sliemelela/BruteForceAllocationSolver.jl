using BruteForceAllocationSolver
using FastGaussQuadrature
using LinearAlgebra
using Statistics
using Interpolations

println("Setting up Problem 3 (Incomplete Market) Numerical Solver...")

# ==============================================================================
# 1. Parameters
# ==============================================================================
M, dt, γ = 10, 1.0, 2.0
u(x) = (x^(1 - γ)) / (1 - γ)
inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# Economic Parameters
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π = 0.05, 0.02, 0.02
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N = 10.0 # Nominal bond maturity

# Correlation Matrix
ρ_rπ, ρ_rS, ρ_πS = 0.5, 0.5, 0.5
ρ_mat = [
    1.0   ρ_rπ  ρ_rS;
    ρ_rπ  1.0   ρ_πS;
    ρ_rS  ρ_πS  1.0
]

# ==============================================================================
# 2. Grids and Integration Nodes
# ==============================================================================
# In Problem 3, the state space is FINANCIAL WEALTH (F_t), not Total Wealth.
G_f = 150
F_grid = generate_log_spaced_grid(10.0, 300.0, G_f)

Z_grids = [
    generate_linear_grid(-0.02, 0.06, 5),  # r_grid
    generate_linear_grid(-0.06, 0.10, 5)   # π_grid
]

# Portfolio weights: ω = [ω_N, ω_S] (NO Inflation-Linked Bond)
omega_space = Vector{Float64}[]
for w_N in range(-2.0, 4.0, length=21)     # Expanded to allow heavier borrowing/leveraging
    for w_S in range(0.0, 1.5, length=11)
        push!(omega_space, [w_N, w_S])
    end
end

# 3D Quadrature Nodes for Expectations
ε_nodes, W_weights = generate_gaussian_shocks(3, 3, ρ_mat)


# ==============================================================================
# 3. Custom Transition Function (2 Risky Assets)
# ==============================================================================
function make_problem3_transition(κ_r, θ_r, σ_r, λ_r, τ_N, κ_π, θ_π, σ_π, λ_S, σ_S, dt)
    B_r_N = abs(κ_r) < 1e-8 ? τ_N : (1.0 - exp(-κ_r * τ_N)) / κ_r
    vol_N_r = -B_r_N * σ_r
    var_N = vol_N_r^2
    var_S = σ_S^2

    return function(Z::Vector{Float64}, ε::Vector{Float64})
        r_n, π_n = Z[1], Z[2]
        ε_r, ε_π, ε_S = ε[1], ε[2], ε[3]

        # 1. State Transitions (Clamped to prevent explosion outside grid)
        r_next = clamp(r_n + κ_r * (θ_r - r_n) * dt + σ_r * sqrt(dt) * ε_r, -0.02, 0.06)
        π_next = clamp(π_n + κ_π * (θ_π - π_n) * dt + σ_π * sqrt(dt) * ε_π, -0.06, 0.10)
        Z_next = [r_next, π_next]

        # 2. Asset Returns (Nominal Bond and Stock only)
        Rf_nom = exp(r_n * dt)

        drift_N = r_n - λ_r * σ_r * B_r_N
        R_N = exp((drift_N - 0.5 * var_N) * dt + vol_N_r * sqrt(dt) * ε_r)

        drift_S = r_n + λ_S * σ_S
        R_S = exp((drift_S - 0.5 * var_S) * dt + σ_S * sqrt(dt) * ε_S)

        # 3. Excess Returns and Base Real Return
        Re = [R_N - Rf_nom, R_S - Rf_nom]
        R_base_real = exp((r_n - π_n) * dt)

        return Z_next, Re, R_base_real
    end
end

transition_prob3 = make_problem3_transition(
    κ_r, overline_r, σ_r, λ_r, τ_N,
    κ_π, overline_π, σ_π, λ_S, σ_S, dt
)

# ==============================================================================
# 4. Budget Constraint & DP Execution
# ==============================================================================
# Human capital is NOT spanned. It is modeled as an additive real cash flow.
function problem3_budget_constraint(F, c, ω, R_e, R_base)
    income_real = 1.0 * dt
    # W_next = Financial return + new income. (c=0 for pure terminal problem)
    F_next = F * (dot(ω, R_e) + R_base) + income_real
    return max(F_next, 1e-10)
end

crra_ex = make_crra_extrapolator(F_grid[1], F_grid[end], γ)

println("Solving Dynamic Program (Pure Terminal Wealth, Unspanned Income)...")
V, pol_w = solve_dynamic_program(
    F_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition_prob3,
    M, u, identity, problem3_budget_constraint, crra_ex
)

# ==============================================================================
# 5. Extract Certainty Equivalent
# ==============================================================================
# To match the Total Wealth baseline (149.80) from Problem 1, we start with
# Financial Wealth of exactly 140.0, because the PV of the income stream is ~9.80.
F_0 = 140.0
r_0 = 0.02
π_0 = 0.02

# Interpolate the Value Function at t=1 for our exact initial state
V_interp = linear_interpolation((F_grid, Z_grids[1], Z_grids[2]), V[:, :, :, 1], extrapolation_bc=Line())
V_0 = V_interp(F_0, r_0, π_0)

CE_0 = calculate_certainty_equivalent(V_0, inv_u)

println("==================================================")
println("Problem 3 (Incomplete Market) Results at t=0:")
println("  Initial Financial Wealth (F_0): ", F_0)
println("  Implied Total Wealth (W_0):     ~149.80")
println("--------------------------------------------------")
println("  Numerical Expected Utility: ", round(V_0, digits=6))
println("  Numerical CE:               ", round(CE_0, digits=4))
println("==================================================")