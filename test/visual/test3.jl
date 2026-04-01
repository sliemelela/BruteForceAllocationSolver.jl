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


# ==============================================================================
# 7. Forward Monte Carlo Simulation (Problem 3 - Incomplete Market)
# ==============================================================================
println("\nRunning Forward Monte Carlo Simulation...")

# Dummy consumption policy for interpolator factory
dummy_pol_c = zeros(size(pol_w))
_, interp_w = create_policy_interpolators(dummy_pol_c, pol_w, F_grid, Z_grids)

# Helper to extract 2 controls
function extract_controls_prob3(F_paths, r_paths, pi_paths, interp_w, dt)
    sims, steps = size(F_paths)
    wN_sim, wS_sim = zeros(sims, steps), zeros(sims, steps)
    for n in 1:steps
        idx = min(length(interp_w), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            F_val = max(F_paths[i, n], 1e-5)
            r, π_val = r_paths[i, n], pi_paths[i, n]
            wN_sim[i, n] = interp_w[idx][1](F_val, r, π_val)
            wS_sim[i, n] = interp_w[idx][2](F_val, r, π_val)
        end
    end
    return wN_sim, wS_sim
end

rate_proc = VasicekProcess(:r, κ_r, overline_r, σ_r, r_0, 1)
pi_proc   = VasicekProcess(:pi, κ_π, overline_π, σ_π, π_0, 2)

B_r_N = (1.0 - exp(-κ_r * τ_N)) / κ_r

# SDE for Financial Real Wealth (WITH additive income)
drift_F3(t, F_val, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    F_safe = max(F_val, 1e-5)
    ω_N = interp_w[idx][1](F_safe, r_val, pi_val)
    ω_S = interp_w[idx][2](F_safe, r_val, pi_val)

    RP_N = ω_N * (-λ_r * σ_r * B_r_N)
    RP_S = ω_S * (λ_S * σ_S)

    return F_safe * (RP_N + RP_S + r_val - pi_val) + 1.0
end

diff_F3(t, F_val, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    F_safe = max(F_val, 1e-5)
    ω_N = interp_w[idx][1](F_safe, r_val, pi_val)
    ω_S = interp_w[idx][2](F_safe, r_val, pi_val)

    return [-F_safe * ω_N * B_r_N * σ_r, 0.0, F_safe * ω_S * σ_S]
end

w3_proc = GenericSDEProcess(:F, drift_F3, diff_F3, F_0, [1, 2, 3], [:r, :pi])

conf_prob3 = MarketConfig(sims=500, T=10.0, dt=1.0, M=10, processes=[rate_proc, pi_proc, w3_proc], correlations=ρ_mat)
world_3 = build_world(conf_prob3)

wN_sim, wS_sim = extract_controls_prob3(world_3.paths.F, world_3.paths.r, world_3.paths.pi, interp_w, 1.0)

# ==============================================================================
# 6. Generating Plots (Problem 3)
# ==============================================================================
println("Generating and saving plots...")

# Identify the fixed indices and explicitly extract their values for the titles
fixed_f_idx = div(G_f, 2)
fixed_F_val = round(F_grid[fixed_f_idx], digits=2)
fixed_r_idx = 3  # Corresponds to r = 0.02
fixed_pi_idx = 3 # Corresponds to π = 0.02

labels_time = ["t = 1", "t = 5", "t = 10"]

# ---------------------------------------------------------
# Plot 1: Value Function
# ---------------------------------------------------------
fig_v = plot_curves(F_grid, [V[:, fixed_r_idx, fixed_pi_idx, 1], V[:, fixed_r_idx, fixed_pi_idx, 5], V[:, fixed_r_idx, fixed_pi_idx, 10]], labels_time;
                    title="Prob 3: Expected Utility V(F) (r=0.02, π=0.02)", xlabel="Financial Wealth (F)", ylabel="Utility", legend_pos=:rb)
save("prob3_term_value_function.png", fig_v)

# ---------------------------------------------------------
# Plot 2: CE Progression
# ---------------------------------------------------------
time_axis = 1:M
ce_over_time = [calculate_certainty_equivalent(V[fixed_f_idx, fixed_r_idx, fixed_pi_idx, t], inv_u) for t in time_axis]
fig_ce_time = plot_curves(time_axis, [ce_over_time], ["CE Financial Wealth"];
                          title="Prob 3: CE Progression (F=$fixed_F_val, r=0.02, π=0.02)", xlabel="Time Step (t)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rt)
save("prob3_term_ce_progression.png", fig_ce_time)

# ---------------------------------------------------------
# Plot 3-4: Mean Strategy Allocations
# ---------------------------------------------------------
fig_mean_wN = plot_mean_with_bounds(wN_sim; title="Mean Nominal Bond Allocation", ylabel="Weight", color=:blue)
save("prob3_term_mean_nominal_bond.png", fig_mean_wN)

fig_mean_wS = plot_mean_with_bounds(wS_sim; title="Mean Stock Allocation", ylabel="Weight", color=:green)
save("prob3_term_mean_stock.png", fig_mean_wS)

# ---------------------------------------------------------
# Plot 5-6: Heatmaps (Interest Rate vs Inflation) at Fixed Wealth
# ---------------------------------------------------------
slice_N_r_vs_pi = [pol_w[fixed_f_idx, r, pi, 1][1] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]
slice_S_r_vs_pi = [pol_w[fixed_f_idx, r, pi, 1][2] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]

fig_heat_N_r_pi = plot_heatmap(Z_grids[1], Z_grids[2], slice_N_r_vs_pi;
                               title="Prob 3: Nominal Bond Policy (F=$fixed_F_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                               colormap=:plasma, label="Weight")
save("prob3_term_heatmap_N_rate_vs_inflation.png", fig_heat_N_r_pi)

fig_heat_S_r_pi = plot_heatmap(Z_grids[1], Z_grids[2], slice_S_r_vs_pi;
                               title="Prob 3: Stock Policy (F=$fixed_F_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                               colormap=:plasma, label="Weight")
save("prob3_term_heatmap_S_rate_vs_inflation.png", fig_heat_S_r_pi)

# ---------------------------------------------------------
# Plot 7-8: Heatmaps (Wealth vs Interest Rate) at Fixed Inflation
# ---------------------------------------------------------
slice_N_F_vs_r = [pol_w[f, r, fixed_pi_idx, 1][1] for f in 1:length(F_grid), r in 1:length(Z_grids[1])]
slice_S_F_vs_r = [pol_w[f, r, fixed_pi_idx, 1][2] for f in 1:length(F_grid), r in 1:length(Z_grids[1])]

fig_heat_N_F_r = plot_heatmap(F_grid, Z_grids[1], slice_N_F_vs_r;
                               title="Prob 3: Nominal Bond Policy (π=0.02, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)",
                               colormap=:viridis, label="Weight")
save("prob3_term_heatmap_N_wealth_vs_rate.png", fig_heat_N_F_r)

fig_heat_S_F_r = plot_heatmap(F_grid, Z_grids[1], slice_S_F_vs_r;
                               title="Prob 3: Stock Policy (π=0.02, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)",
                               colormap=:viridis, label="Weight")
save("prob3_term_heatmap_S_wealth_vs_rate.png", fig_heat_S_F_r)

# ---------------------------------------------------------
# Plot 9: Wealth vs Human Capital Evolution
# ---------------------------------------------------------
times_seq = 1:M
H_mean = [(M - t + 1) * dt for t in times_seq] # Approx Human Capital assuming real rate ≈ 0
F_mean = vec(mean(world_3.paths.F, dims=1))[1:M]

fig_wealth_comp = plot_curves(times_seq, [F_mean, H_mean], ["Financial Wealth (F)", "Human Capital (H)"];
                              title="Wealth Composition Over Time", xlabel="Time (Steps)", ylabel="Value", legend_pos=:rc)
save("prob3_term_wealth_composition.png", fig_wealth_comp)

println("All Incomplete Market solutions, simulations, and plots generated successfully!")