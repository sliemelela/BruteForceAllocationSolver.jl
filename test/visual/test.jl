using BruteForceAllocationSolver
using FinancialMarketSimulation
using FastGaussQuadrature
using CairoMakie
using LinearAlgebra
using Statistics
using Interpolations

println("==================================================")
println("Setting up Problem 1 (Complete Market) Solver...")
println("==================================================")

# ==============================================================================
# 1. Parameters & Market Prices of Risk
# ==============================================================================
M, dt, γ = 10, 1.0, 2.0
T = M * dt
γ_tilde = (γ - 1.0) / γ

u(x) = (x^(1 - γ)) / (1 - γ)
inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# Economic Parameters
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π, λ_π = 0.05, 0.02, 0.02, 0.05
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N = 10.0 # Nominal bond maturity
τ_I = 10.0 # Inflation-linked bond maturity

# Correlation Matrix
ρ_rπ, ρ_rS, ρ_πS = 0.5, 0.5, 0.5
ρ_mat = [
    1.0   ρ_rπ  ρ_rS;
    ρ_rπ  1.0   ρ_πS;
    ρ_rS  ρ_πS  1.0
]

# Market prices of risk (λ = φ * ρ  =>  φ = ρ \ λ)
lambda_vec = [λ_r, λ_π, λ_S]
phi_vec = -(ρ_mat \ lambda_vec)

println("Market Prices of Risk (λ): ", lambda_vec)
println("Factor loadings (φ):       ", phi_vec)


# ==============================================================================
# 2. Exact Analytical Baseline (using Lemma A.7)
# ==============================================================================
println("\nCalculating Analytical Baseline...")

B_r(h) = abs(κ_r) < 1e-8 ? h : (1.0 - exp(-κ_r * h)) / κ_r
B_π(h) = abs(κ_π) < 1e-8 ? h : (1.0 - exp(-κ_π * h)) / κ_π

function E2(h)
    phi_norm_sq = dot(phi_vec, ρ_mat * phi_vec)
    term1 = h * (overline_π - overline_r - 0.5 * phi_norm_sq)
    term2 = B_r(h) * overline_r
    term3 = -B_π(h) * overline_π
    return term1 + term2 + term3
end

function V2(h)
    phi_norm_sq = dot(phi_vec, ρ_mat * phi_vec)
    term1 = (σ_r^2) / (2 * κ_r^3) * (2 * exp(-κ_r * h) - 0.5 * exp(-2 * κ_r * h) - 1.5)
    term2 = (σ_π^2) / (2 * κ_π^3) * (2 * exp(-κ_π * h) - 0.5 * exp(-2 * κ_π * h) - 1.5)
    term3 = h * (0.5 * phi_norm_sq + (σ_r^2) / (2 * κ_r^2) + (σ_π^2) / (2 * κ_π^2))
    term4 = (σ_r / κ_r) * (phi_vec[1] + ρ_rπ * phi_vec[2] + ρ_rS * phi_vec[3]) * (B_r(h) - h)
    term5 = -(σ_π / κ_π) * (phi_vec[2] + ρ_rπ * phi_vec[1] + ρ_πS * phi_vec[3]) * (B_π(h) - h)
    term6 = ρ_rπ * (σ_r * σ_π) / (κ_r * κ_π) * (B_r(h) + B_π(h) - (1.0 - exp(-(κ_r + κ_π) * h)) / (κ_r + κ_π) - h)
    return term1 + term2 + term3 + term4 + term5 + term6
end

function A_I(h)
    term1 = overline_r * (B_r(h) - h) - overline_π * (B_π(h) - h)
    term2 = (σ_r^2 / (2 * κ_r^3)) * (2 * exp(-κ_r * h) - 0.5 * exp(-2 * κ_r * h) - 1.5)
    term3 = (σ_π^2 / (2 * κ_π^3)) * (2 * exp(-κ_π * h) - 0.5 * exp(-2 * κ_π * h) - 1.5)
    term4 = h * (σ_r^2 / (2 * κ_r^2) + σ_π^2 / (2 * κ_π^2))
    term5 = (σ_r / κ_r) * (phi_vec[1] + ρ_rπ * phi_vec[2] + ρ_rS * phi_vec[3]) * (B_r(h) - h)
    term6 = -(σ_π / κ_π) * (phi_vec[2] + ρ_rπ * phi_vec[1] + ρ_πS * phi_vec[3]) * (B_π(h) - h)
    term7 = ρ_rπ * (σ_r * σ_π) / (κ_r * κ_π) * (B_r(h) + B_π(h) - (1.0 - exp(-(κ_r + κ_π) * h)) / (κ_r + κ_π) - h)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7
end

function exact_human_capital_lemma_A7(M, dt, r_0, π_0)
    H_0_exact = 0.0
    Pi_0 = 1.0
    for j in 1:M
        tau = j * dt
        P_I = Pi_0 * exp(A_I(tau) - B_r(tau) * r_0 + B_π(tau) * π_0)
        income = 1.0 * dt
        H_0_exact += income * P_I
    end
    return H_0_exact
end

# Initial state variables
r_0 = 0.02
π_0 = 0.02

# Evaluate Total Wealth
F_0 = 140.0
H_0 = exact_human_capital_lemma_A7(M, dt, r_0, π_0)
W_0 = F_0 + H_0

# Closed-form Analytical CE
exponent = B_r(T) * r_0 - B_π(T) * π_0 - E2(T) - γ_tilde * V2(T)
analytical_CE = W_0 * exp(exponent)
analytical_V0 = (W_0^(1 - γ)) / (1 - γ) * exp((γ - 1) * (-exponent))

println("  Exact Human Capital (H_0): ", round(H_0, digits=4))
println("  Initial Total Wealth (W_0): ", round(W_0, digits=4))
println("  Analytical Expected Utility: ", round(analytical_V0, digits=6))
println("  Analytical CE:               ", round(analytical_CE, digits=4))


# ==============================================================================
# 3. Numerical DP Setup (Complete Market - 3 Assets, Total Wealth)
# ==============================================================================
println("\nSetting up Numerical Grids...")

G_w = 150
W_grid = generate_log_spaced_grid(10.0, 300.0, G_w)

Z_grids = [
    generate_linear_grid(-0.02, 0.06, 5),  # r_grid
    generate_linear_grid(-0.06, 0.10, 5)   # π_grid
]

# Portfolio weights: ω = [ω_N, ω_I, ω_S]
omega_space = Vector{Float64}[]
for w_N in range(-1.0, 2.0, length=11)
    for w_I in range(-1.0, 2.0, length=11)
        for w_S in range(0.0, 1.5, length=11)
            push!(omega_space, [w_N, w_I, w_S])
        end
    end
end

ε_nodes, W_weights = generate_gaussian_shocks(3, 3, ρ_mat)

function make_problem1_transition(κ_r, θ_r, σ_r, λ_r, τ_N, κ_π, θ_π, σ_π, λ_π, τ_I, λ_S, σ_S, ρ_rπ, dt)
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

        r_next = clamp(r_n + κ_r * (θ_r - r_n) * dt + σ_r * sqrt(dt) * ε_r, -0.02, 0.06)
        π_next = clamp(π_n + κ_π * (θ_π - π_n) * dt + σ_π * sqrt(dt) * ε_π, -0.06, 0.10)
        Z_next = [r_next, π_next]

        Rf_nom = exp(r_n * dt)

        drift_N = r_n - λ_r * σ_r * B_r_N
        R_N = exp((drift_N - 0.5 * var_N) * dt + vol_N_r * sqrt(dt) * ε_r)

        drift_I = r_n - λ_r * σ_r * B_r_I + λ_π * σ_π * B_π_I
        R_I = exp((drift_I - 0.5 * var_I) * dt + vol_I_r * sqrt(dt) * ε_r + vol_I_π * sqrt(dt) * ε_π)

        drift_S = r_n + λ_S * σ_S
        R_S = exp((drift_S - 0.5 * var_S) * dt + σ_S * sqrt(dt) * ε_S)

        Re = [R_N - Rf_nom, R_I - Rf_nom, R_S - Rf_nom]
        R_base_real = exp((r_n - π_n) * dt)

        return Z_next, Re, R_base_real
    end
end

transition_prob1 = make_problem1_transition(κ_r, overline_r, σ_r, λ_r, τ_N, κ_π, overline_π, σ_π, λ_π, τ_I, λ_S, σ_S, ρ_rπ, dt)

function problem1_budget_constraint(W, c, ω, R_e, R_base)
    W_next = W * (dot(ω, R_e) + R_base)
    return max(W_next, 1e-10)
end
crra_ex = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

# ==============================================================================
# 4. DP Execution & Numerical Baseline
# ==============================================================================
println("Solving Dynamic Program (Pure Terminal Wealth)...")
V, pol_w = solve_dynamic_program(
    W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition_prob1,
    M, u, identity, problem1_budget_constraint, crra_ex
)

# Extract Numerical CE using W_0 calculated from analytical step
V_interp = linear_interpolation((W_grid, Z_grids[1], Z_grids[2]), V[:, :, :, 1], extrapolation_bc=Line())
V_0_num = V_interp(W_0, r_0, π_0)
CE_0_num = calculate_certainty_equivalent(V_0_num, inv_u)

println("\n==================================================")
println("Problem 1 (Complete Market) Final Comparison:")
println("  Total Initial Wealth (W_0): ", round(W_0, digits=4))
println("  Analytical CE:              ", round(analytical_CE, digits=4))
println("  Numerical CE:               ", round(CE_0_num, digits=4))
println("  Discretization Friction:    ", round(analytical_CE - CE_0_num, digits=4))
println("==================================================")


# ==============================================================================
# 5. Forward Monte Carlo Simulation
# ==============================================================================
println("\nRunning Forward Monte Carlo Simulation...")

dummy_pol_c = zeros(size(pol_w))
_, interp_w = create_policy_interpolators(dummy_pol_c, pol_w, W_grid, Z_grids)

function extract_controls_prob1(W_paths, r_paths, pi_paths, interp_w, dt)
    sims, steps = size(W_paths)
    wN_sim, wI_sim, wS_sim = zeros(sims, steps), zeros(sims, steps), zeros(sims, steps)
    for n in 1:steps
        idx = min(length(interp_w), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            W = max(W_paths[i, n], 1e-5)
            r, π_val = r_paths[i, n], pi_paths[i, n]
            wN_sim[i, n] = interp_w[idx][1](W, r, π_val)
            wI_sim[i, n] = interp_w[idx][2](W, r, π_val)
            wS_sim[i, n] = interp_w[idx][3](W, r, π_val)
        end
    end
    return wN_sim, wI_sim, wS_sim
end

rate_proc = VasicekProcess(:r, κ_r, overline_r, σ_r, r_0, 1)
pi_proc   = VasicekProcess(:pi, κ_π, overline_π, σ_π, π_0, 2)

B_r_N = (1.0 - exp(-κ_r * τ_N)) / κ_r
B_r_I = (1.0 - exp(-κ_r * τ_I)) / κ_r
B_pi_I = (1.0 - exp(-κ_π * τ_I)) / κ_π

drift_W1(t, W, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    W_safe = max(W, 1e-5)
    ω_N = interp_w[idx][1](W_safe, r_val, pi_val)
    ω_I = interp_w[idx][2](W_safe, r_val, pi_val)
    ω_S = interp_w[idx][3](W_safe, r_val, pi_val)

    RP_N = ω_N * (-λ_r * σ_r * B_r_N)
    RP_I = ω_I * (-λ_r * σ_r * B_r_I + λ_π * σ_π * B_pi_I)
    RP_S = ω_S * (λ_S * σ_S)

    return W_safe * (RP_N + RP_I + RP_S + r_val - pi_val)
end

diff_W1(t, W, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    W_safe = max(W, 1e-5)
    ω_N = interp_w[idx][1](W_safe, r_val, pi_val)
    ω_I = interp_w[idx][2](W_safe, r_val, pi_val)
    ω_S = interp_w[idx][3](W_safe, r_val, pi_val)

    diff_r = W_safe * (-ω_N * B_r_N * σ_r - ω_I * B_r_I * σ_r)
    diff_pi = W_safe * (ω_I * B_pi_I * σ_π)
    diff_S = W_safe * (ω_S * σ_S)
    return [diff_r, diff_pi, diff_S]
end

w1_proc = GenericSDEProcess(:W, drift_W1, diff_W1, W_0, [1, 2, 3], [:r, :pi])

conf_prob1 = MarketConfig(sims=500, T=T, dt=dt, M=M, processes=[rate_proc, pi_proc, w1_proc], correlations=ρ_mat)
world_1 = build_world(conf_prob1)

wN_sim, wI_sim, wS_sim = extract_controls_prob1(world_1.paths.W, world_1.paths.r, world_1.paths.pi, interp_w, dt)


# ==============================================================================
# 6. Generating Plots (Problem 1)
# ==============================================================================
println("Generating and saving plots...")

# Identify the fixed indices and explicitly extract their values for the titles
fixed_w_idx = div(G_w, 2)
fixed_W_val = round(W_grid[fixed_w_idx], digits=2)
fixed_r_idx = 3  # Corresponds to r = 0.02
fixed_pi_idx = 3 # Corresponds to π = 0.02

labels_time = ["t = 1", "t = 5", "t = 10"]

# ---------------------------------------------------------
# Plot 1: Value Function
# ---------------------------------------------------------
fig_v = plot_curves(W_grid, [V[:, fixed_r_idx, fixed_pi_idx, 1], V[:, fixed_r_idx, fixed_pi_idx, 5], V[:, fixed_r_idx, fixed_pi_idx, 10]], labels_time;
                    title="Prob 1: Expected Utility V(W) (r=0.02, π=0.02)", xlabel="Total Real Wealth (W)", ylabel="Utility", legend_pos=:rb)
save("prob1_term_value_function.png", fig_v)

# ---------------------------------------------------------
# Plot 2: CE Progression
# ---------------------------------------------------------
time_axis = 1:M
ce_over_time = [calculate_certainty_equivalent(V[fixed_w_idx, fixed_r_idx, fixed_pi_idx, t], inv_u) for t in time_axis]
fig_ce_time = plot_curves(time_axis, [ce_over_time], ["CE Total Wealth"];
                          title="Prob 1: CE Progression (W=$fixed_W_val, r=0.02, π=0.02)", xlabel="Time Step (t)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rt)
save("prob1_term_ce_progression.png", fig_ce_time)

# ---------------------------------------------------------
# Plot 3-5: Mean Strategy Allocations
# ---------------------------------------------------------
fig_mean_wN = plot_mean_with_bounds(wN_sim; title="Mean Nominal Bond Allocation", ylabel="Weight", color=:blue)
save("prob1_term_mean_nominal_bond.png", fig_mean_wN)

fig_mean_wI = plot_mean_with_bounds(wI_sim; title="Mean ILB Allocation", ylabel="Weight", color=:purple)
save("prob1_term_mean_ilb.png", fig_mean_wI)

fig_mean_wS = plot_mean_with_bounds(wS_sim; title="Mean Stock Allocation", ylabel="Weight", color=:green)
save("prob1_term_mean_stock.png", fig_mean_wS)

# ---------------------------------------------------------
# Plot 6-8: Heatmaps (Interest Rate vs Inflation) at Fixed Wealth
# ---------------------------------------------------------
slice_N_r_vs_pi = [pol_w[fixed_w_idx, r, pi, 1][1] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]
slice_I_r_vs_pi = [pol_w[fixed_w_idx, r, pi, 1][2] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]
slice_S_r_vs_pi = [pol_w[fixed_w_idx, r, pi, 1][3] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]

fig_heat_N_r_pi = plot_heatmap(Z_grids[1], Z_grids[2], slice_N_r_vs_pi;
                               title="Prob 1: Nominal Bond Policy (W=$fixed_W_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                               colormap=:plasma, label="Weight")
save("prob1_term_heatmap_N_rate_vs_inflation.png", fig_heat_N_r_pi)

fig_heat_I_r_pi = plot_heatmap(Z_grids[1], Z_grids[2], slice_I_r_vs_pi;
                               title="Prob 1: ILB Policy (W=$fixed_W_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                               colormap=:plasma, label="Weight")
save("prob1_term_heatmap_I_rate_vs_inflation.png", fig_heat_I_r_pi)

fig_heat_S_r_pi = plot_heatmap(Z_grids[1], Z_grids[2], slice_S_r_vs_pi;
                               title="Prob 1: Stock Policy (W=$fixed_W_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                               colormap=:plasma, label="Weight")
save("prob1_term_heatmap_S_rate_vs_inflation.png", fig_heat_S_r_pi)

# ---------------------------------------------------------
# Plot 9-11: Heatmaps (Wealth vs Interest Rate) at Fixed Inflation
# ---------------------------------------------------------
slice_N_W_vs_r = [pol_w[w, r, fixed_pi_idx, 1][1] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]
slice_I_W_vs_r = [pol_w[w, r, fixed_pi_idx, 1][2] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]
slice_S_W_vs_r = [pol_w[w, r, fixed_pi_idx, 1][3] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]

fig_heat_N_W_r = plot_heatmap(W_grid, Z_grids[1], slice_N_W_vs_r;
                               title="Prob 1: Nominal Bond Policy (π=0.02, t=1)", xlabel="Total Real Wealth (W)", ylabel="Interest Rate (r)",
                               colormap=:viridis, label="Weight")
save("prob1_term_heatmap_N_wealth_vs_rate.png", fig_heat_N_W_r)

fig_heat_I_W_r = plot_heatmap(W_grid, Z_grids[1], slice_I_W_vs_r;
                               title="Prob 1: ILB Policy (π=0.02, t=1)", xlabel="Total Real Wealth (W)", ylabel="Interest Rate (r)",
                               colormap=:viridis, label="Weight")
save("prob1_term_heatmap_I_wealth_vs_rate.png", fig_heat_I_W_r)

fig_heat_S_W_r = plot_heatmap(W_grid, Z_grids[1], slice_S_W_vs_r;
                               title="Prob 1: Stock Policy (π=0.02, t=1)", xlabel="Total Real Wealth (W)", ylabel="Interest Rate (r)",
                               colormap=:viridis, label="Weight")
save("prob1_term_heatmap_S_wealth_vs_rate.png", fig_heat_S_W_r)

println("All Complete Market solutions, simulations, and plots generated successfully!")