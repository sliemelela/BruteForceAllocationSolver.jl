using BruteForceAllocationSolver
using FastGaussQuadrature
using LinearAlgebra
using Statistics
using Interpolations

println("Setting up Problem 1 (Complete Market) Numerical Solver...")

# ==============================================================================
# 1. Parameters
# ==============================================================================
M, dt, γ = 10, 1.0, 2.0
u(x) = (x^(1 - γ)) / (1 - γ)
inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# Economic Parameters (Aligned with Problem 3 for an apples-to-apples comparison)
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π, λ_π = 0.05, 0.02, 0.02, 0.05 # Added an inflation risk premium
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N = 10.0 # Nominal bond maturity
τ_I = 10.0 # Inflation-linked bond (ILB) maturity

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
# Since we are evaluating Total Wealth (F_t + H_t), the wealth bounds need to be
# larger than in Problem 3 (where state space was just F_t).

G_w = 150
W_grid = generate_log_spaced_grid(10.0, 300.0, G_w)

Z_grids = [
    generate_linear_grid(-0.02, 0.06, 5),  # r_grid  (covers ±4σ)
    generate_linear_grid(-0.06, 0.10, 5)   # π_grid  (covers ±4σ)
]

# Portfolio weights: ω = [ω_N, ω_I, ω_S]
# Because this is a complete market solving on total wealth, the optimal weights
# are constant. We don't need a huge grid, just one that covers the analytical solution.
omega_space = Vector{Float64}[]
for w_N in range(-1.0, 2.0, length=11)
    for w_I in range(-1.0, 2.0, length=11)
        for w_S in range(0.0, 1.5, length=11)
            push!(omega_space, [w_N, w_I, w_S])
        end
    end
end

# 3D Quadrature Nodes for Expectations
ε_nodes, W_weights = generate_gaussian_shocks(3, 3, ρ_mat)


# ==============================================================================
# 3. Custom Transition Function (3 Risky Assets)
# ==============================================================================
function make_problem1_transition(κ_r, θ_r, σ_r, λ_r, τ_N, κ_π, θ_π, σ_π, λ_π, τ_I, λ_S, σ_S, ρ_rπ, dt)
    B_r(τ) = abs(κ_r) < 1e-8 ? τ : (1.0 - exp(-κ_r * τ)) / κ_r
    B_π(τ) = abs(κ_π) < 1e-8 ? τ : (1.0 - exp(-κ_π * τ)) / κ_π

    B_r_N = B_r(τ_N)
    B_r_I = B_r(τ_I)
    B_π_I = B_π(τ_I)

    vol_N_r = -B_r_N * σ_r
    vol_I_r = -B_r_I * σ_r
    vol_I_π = B_π_I * σ_π

    var_N = vol_N_r^2
    var_I = vol_I_r^2 + vol_I_π^2 + 2 * ρ_rπ * vol_I_r * vol_I_π
    var_S = σ_S^2

    return function(Z::Vector{Float64}, ε::Vector{Float64})
        r_n, π_n = Z[1], Z[2]
        ε_r, ε_π, ε_S = ε[1], ε[2], ε[3]

        # 1. State Transitions (NOW CLAMPED TO GRID BOUNDS)
        r_next = clamp(r_n + κ_r * (θ_r - r_n) * dt + σ_r * sqrt(dt) * ε_r, -0.02, 0.06)
        π_next = clamp(π_n + κ_π * (θ_π - π_n) * dt + σ_π * sqrt(dt) * ε_π, -0.06, 0.10)
        Z_next = [r_next, π_next]

        # 2. Asset Returns
        Rf_nom = exp(r_n * dt)

        drift_N = r_n - λ_r * σ_r * B_r_N
        R_N = exp((drift_N - 0.5 * var_N) * dt + vol_N_r * sqrt(dt) * ε_r)

        drift_I = r_n - λ_r * σ_r * B_r_I + λ_π * σ_π * B_π_I
        R_I = exp((drift_I - 0.5 * var_I) * dt + vol_I_r * sqrt(dt) * ε_r + vol_I_π * sqrt(dt) * ε_π)

        drift_S = r_n + λ_S * σ_S
        R_S = exp((drift_S - 0.5 * var_S) * dt + σ_S * sqrt(dt) * ε_S)

        # 3. Excess Returns and Base Return
        Re = [R_N - Rf_nom, R_I - Rf_nom, R_S - Rf_nom]
        R_base_real = exp((r_n - π_n) * dt)

        return Z_next, Re, R_base_real
    end
end

transition_prob1 = make_problem1_transition(
    κ_r, overline_r, σ_r, λ_r, τ_N,
    κ_π, overline_π, σ_π, λ_π, τ_I,
    λ_S, σ_S, ρ_rπ, dt
)

# ==============================================================================
# 4. Budget Constraint & DP Execution
# ==============================================================================
# Because human capital is fully spanned, we treat it as tradeable financial wealth.
# Total Wealth = Financial Wealth + Human Capital. Outside income is 0.0.
function problem1_budget_constraint(W, c, ω, R_e, R_base)
    W_next = (1.0 - c) * W * (dot(ω, R_e) + R_base)
    return max(W_next, 1e-10) # Hard floor to prevent bankruptcy flipping signs
end
crra_ex = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

println("Solving Dynamic Program (Pure Terminal Wealth)...")
# Note: We use the terminal wealth solver signature (no c_grid, no β)
V, pol_w = solve_dynamic_program(
    W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition_prob1,
    M, u, identity, problem1_budget_constraint, crra_ex
)

# ==============================================================================
# 5. Extract Certainty Equivalent
# ==============================================================================
# Let's assume initial financial wealth F_0 = 100, and human capital H_0 = 50.
# Therefore, initial Total Real Wealth W_0 = 150.
W_0 = 150.0
r_0 = 0.02
π_0 = 0.02

# Interpolate the Value Function at t=1 for our exact initial state
V_interp = linear_interpolation((W_grid, Z_grids[1], Z_grids[2]), V[:, :, :, 1], extrapolation_bc=Line())
V_0 = V_interp(W_0, r_0, π_0)

CE_0 = calculate_certainty_equivalent(V_0, inv_u)

println("==================================================")
println("Problem 1 (Complete Market) Results at t=0:")
println("  Initial Total Wealth (W_0): ", W_0)
println("  Expected Utility:           ", round(V_0, digits=6))
println("  Numerical CE:               ", round(CE_0, digits=4))
println("==================================================")
println("This Numerical CE represents your grid's baseline performance.")
println("To find the exact numerical friction penalty, subtract this from the Analytical CE.")