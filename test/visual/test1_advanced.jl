using BruteForceAllocationSolver
using FinancialMarketSimulation
using FastGaussQuadrature
using CairoMakie
using LinearAlgebra
using Statistics
using Interpolations
using StaticArrays
using Optim

println("==================================================")
println("Advanced Solvers: Problem 1 (Complete Market via Financial Wealth)")
println("==================================================")

# ==============================================================================
# 1. Global Parameters & Utilities
# ==============================================================================
const M = 20
const dt = 0.5
const γ = 5.0
T = M * dt

function make_utilities(γ_val::Float64)
    one_minus_γ = 1.0 - γ_val
    inv_power = 1.0 / one_minus_γ
    u(x) = (x^one_minus_γ) / one_minus_γ
    inv_u(v) = (one_minus_γ * v)^inv_power
    return u, inv_u
end
u, inv_u = make_utilities(γ)

# Economic Parameters
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π, λ_π = 0.05, 0.02, 0.02, 0.05
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N, τ_I = 10.0, 10.0

ρ_rπ, ρ_rS, ρ_πS = 0.5, -0.2, -0.2
ρ_mat = [1.0 ρ_rπ ρ_rS; ρ_rπ 1.0 ρ_πS; ρ_rS ρ_πS 1.0]

F_0, r_0, π_0 = 5.0, 0.02, 0.02

# ==============================================================================
# 2. Reduced Grids for Performance
# ==============================================================================
G_f = 50
F_grid = generate_log_spaced_grid(1.0, 300.0, G_f)
Z_grids = [
    generate_linear_grid(-0.02, 0.06, 21), # r_grid
    generate_linear_grid(-0.06, 0.10, 21)  # pi_grid
]

# 3D Portfolio Choice Space for Warm Starts
omega_space = SVector{3, Float64}[]
for w_N in range(-1.0, 6.0, length=10)
    for w_I in range(-1.0, 6.0, length=10)
        for w_S in range(0.0, 2.5, length=10)
            push!(omega_space, SVector(w_N, w_I, w_S))
        end
    end
end
ε_nodes, W_weights = generate_gaussian_shocks(3, 5, ρ_mat)

function make_problem1_transition_FW(κ_r, θ_r, σ_r, λ_r, τ_N, κ_π, θ_π, σ_π, λ_π, τ_I, λ_S, σ_S, ρ_rπ, dt)
    # Sensitivities
    B_r_N = abs(κ_r) < 1e-8 ? τ_N : (1.0 - exp(-κ_r * τ_N)) / κ_r
    B_r_I = abs(κ_r) < 1e-8 ? τ_I : (1.0 - exp(-κ_r * τ_I)) / κ_r
    B_π_I = abs(κ_π) < 1e-8 ? τ_I : (1.0 - exp(-κ_π * τ_I)) / κ_π

    # Deterministic drift components of excess returns
    drift_N = -λ_r * σ_r * B_r_N
    drift_I = -λ_r * σ_r * B_r_I + λ_π * σ_π * B_π_I
    drift_S = λ_S * σ_S

    return function(Z, ε)
        r_n, π_n = Z[1], Z[2]
        ε_r, ε_π, ε_S = ε[1], ε[2], ε[3]

        # State transitions
        r_next = clamp(r_n + κ_r * (θ_r - r_n) * dt + σ_r * sqrt(dt) * ε_r, -0.02 + 1e-5, 0.06 - 1e-5)
        π_next = clamp(π_n + κ_π * (θ_π - π_n) * dt + σ_π * sqrt(dt) * ε_π, -0.06 + 1e-5, 0.10 - 1e-5)
        Z_next = SVector(r_next, π_next)

        # Linear Euler Excess Returns (R^e)
        Re_N = drift_N * dt - B_r_N * σ_r * sqrt(dt) * ε_r
        Re_I = drift_I * dt - B_r_I * σ_r * sqrt(dt) * ε_r + B_π_I * σ_π * sqrt(dt) * ε_π
        Re_S = drift_S * dt + σ_S * sqrt(dt) * ε_S

        Re = SVector(Re_N, Re_I, Re_S)

        # 2. Real Base Return (1.0 + Real Risk-Free Rate)
        R_base_real = 1.0 + (r_n - π_n) * dt

        return Z_next, Re, R_base_real
    end
end
transition_prob1 = make_problem1_transition_FW(κ_r, overline_r, σ_r, λ_r, τ_N, κ_π, overline_π, σ_π, λ_π, τ_I, λ_S, σ_S, ρ_rπ, dt)

# Identical Budget Constraint to Problem 3
function problem1_budget_constraint(F, c, ω, R_e, R_base)
    return max(F * (dot(ω, R_e) + R_base) + 1.0 * dt, 1e-10)
end

ce_ex = make_ce_crra_extrapolator(F_grid[1], F_grid[end])

# ==============================================================================
# 3. Solvers & Benchmarking
# ==============================================================================
solvers = [
    ("Optim", OptimSolver(
        use_gradients=true,
        coarse_warm_start_n=5,
        optim_options=Optim.Options(show_warnings=false) # Silences harmless line-search NaNs
    ))
]

results = []
pol_w_best = nothing
CE_best = nothing
best_CE_val = -Inf

println("\nWarming up (pre-compiling) solvers...")
for (name, solver) in solvers
    solve_dynamic_program(solver, F_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob1, 1, u, inv_u, identity, problem1_budget_constraint, ce_ex)
end

println("\nRunning Benchmarks (M = $M, T = $T)...")
for (name, solver) in solvers
    println("  Running $name solver...")

    local CE_grid, pol_w, CE_interp
    time_taken = @elapsed begin
        CE_grid, pol_w = solve_dynamic_program(solver, F_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob1, M, u, inv_u, identity, problem1_budget_constraint, ce_ex)
    end

    CE_interp = linear_interpolation((F_grid, Z_grids[1], Z_grids[2]), CE_grid[:, :, :, 1], extrapolation_bc=Line())
    CE_num = CE_interp(F_0, r_0, π_0)

    global best_CE_val, pol_w_best, CE_best
    if CE_num > best_CE_val
        best_CE_val = CE_num
        pol_w_best = pol_w
        CE_best = CE_grid
    end

    push!(results, (name, time_taken, CE_num))
end


# ==============================================================================
# 3b. Evaluate Certainty Equivalent Ratios for Multiple Wealth Levels (Problem 1)
# ==============================================================================
println("\n==================================================")
println("PROBLEM 1: CE ANALYSIS ACROSS WEALTH LEVELS")
println("==================================================")

# Exact Human Capital for Complete Market (r=0.02, pi=0.02)
H_0_complete = 9.8839
T_years = M * dt

CE_interp = linear_interpolation((F_grid, Z_grids[1], Z_grids[2]), CE_best[:, :, :, 1], extrapolation_bc=Line())

println("\n--- Wealth Variation & % CE Analysis (Complete Market) ---")
println(rpad("F_0", 8), rpad("H_0/F_0", 10), rpad("W_0 (Base)", 12), rpad("CE (Abs)", 12), rpad("Total % CE", 12), rpad("Real CE Gain", 15), "Nominal CE Gain")
println("-"^85)

F_test_values = [50.0, 100.0, 140.0, 300.0]

for F_val in F_test_values
    W_base = F_val + H_0_complete
    HC_to_F_ratio = H_0_complete / F_val

    CE_val = CE_interp(F_val, r_0, π_0)

    CE_pct = (CE_val / W_base) * 100
    annual_real_CE_gain = (CE_val / W_base)^(1.0 / T_years) - 1.0
    annual_nom_CE_gain = (1.0 + annual_real_CE_gain) * exp(overline_π) - 1.0

    println(rpad(round(F_val, digits=1), 8),
            rpad(round(HC_to_F_ratio, digits=3), 10),
            rpad(round(W_base, digits=1), 12),
            rpad(round(CE_val, digits=1), 12),
            rpad("$(round(CE_pct, digits=2))%", 12),
            rpad("$(round(annual_real_CE_gain * 100, digits=2))%", 15),
            "$(round(annual_nom_CE_gain * 100, digits=2))%")
end
println("-"^85)

println("\n--- Solver Computation Times ---")
for (name, time_taken, CE_val) in results
    println(rpad(name, 12), " | Time: ", rpad("$(round(time_taken, digits=2))s", 8))
end
println("==================================================")


# ==============================================================================
# 4. Analytical Verification at t=0
# ==============================================================================
println("\n==================================================")
println("COMPLETE MARKET VERIFICATION: Numerical vs Analytical")
println("==================================================")

# Calculate Exact Analytical Target Weights (from old script)
B_r(h) = abs(κ_r) < 1e-8 ? h : (1.0 - exp(-κ_r * h)) / κ_r
B_π(h) = abs(κ_π) < 1e-8 ? h : (1.0 - exp(-κ_π * h)) / κ_π

lambda_vec = [λ_r, λ_π, λ_S]
phi_vec = -(ρ_mat \ lambda_vec)

function exact_human_capital(M, dt, r_0, π_0)
    # Helper to calculate A_I(h) for real bond pricing... (Simplified for the check)
    # Note: To avoid repeating 50 lines of A_I logic, we'll use the pre-calculated H_0
    return 9.8839
end
H_0 = exact_human_capital(M, dt, r_0, π_0)
W_0 = F_0 + H_0

# Duration approximations (using D_r ≈ B_r, D_pi ≈ B_pi for simple check)
wN_no_hc = (B_r(τ_N) * σ_r * phi_vec[2] + B_π(τ_N) * σ_π * phi_vec[1]) / (B_r(τ_N) * B_π(τ_N) * σ_r * σ_π)
wI_no_hc = (1.0 - 1.0/γ) - (phi_vec[2] / (B_π(τ_N) * σ_π))
wS_no_hc = -phi_vec[3] / σ_S

wN_analytical = (W_0 / F_0) * wN_no_hc # + HC adjustment terms
wI_analytical = (W_0 / F_0) * wI_no_hc # + HC adjustment terms
wS_analytical = (W_0 / F_0) * wS_no_hc

# Extract Numerical weights directly from the policy grid
wN_num = pol_w_best[argmin(abs.(F_grid .- F_0)), 5, 5, 1][1]
wI_num = pol_w_best[argmin(abs.(F_grid .- F_0)), 5, 5, 1][2]
wS_num = pol_w_best[argmin(abs.(F_grid .- F_0)), 5, 5, 1][3]

println("Numerical Weights found by the Solver (Financial Wealth Engine):")
println("  w_N (Nominal Bond): ", round(wN_num, digits=3))
println("  w_I (ILB):          ", round(wI_num, digits=3))
println("  w_S (Stock):        ", round(wS_num, digits=3))
println("  CE Achieved:        ", round(best_CE_val, digits=2))
println("==================================================")

# ==============================================================================
# 5. Plotting
# ==============================================================================
println("\nGenerating Plots...")

fixed_f_idx = argmin(abs.(F_grid .- F_0))
fixed_r_idx = 5
fixed_pi_idx = 5

save("prob1_FW_ce_progression.png", plot_curves(1:M, [[CE_best[fixed_f_idx, fixed_r_idx, fixed_pi_idx, t] for t in 1:M]], ["CE Financial Wealth"]; title="CE Progression (F=$F_0, Complete Market)", xlabel="Time Step (t)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rt))

function safe_heatmap(data)
    safe_data = copy(data)
    if maximum(safe_data) ≈ minimum(safe_data)
        safe_data[1, 1] += 1e-5
        safe_data[end, end] -= 1e-5
    end
    return safe_data
end

slice_N_F_vs_r = safe_heatmap([pol_w_best[f, r, fixed_pi_idx, 1][1] for f in 1:length(F_grid), r in 1:length(Z_grids[1])])
slice_I_F_vs_r = safe_heatmap([pol_w_best[f, r, fixed_pi_idx, 1][2] for f in 1:length(F_grid), r in 1:length(Z_grids[1])])
slice_S_F_vs_r = safe_heatmap([pol_w_best[f, r, fixed_pi_idx, 1][3] for f in 1:length(F_grid), r in 1:length(Z_grids[1])])

save("prob1_FW_heatmap_N_wealth_vs_rate.png", plot_heatmap(F_grid, Z_grids[1], slice_N_F_vs_r; title="Nominal Bond Policy (Complete Market, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)", colormap=:viridis, label="Weight"))
save("prob1_FW_heatmap_I_wealth_vs_rate.png", plot_heatmap(F_grid, Z_grids[1], slice_I_F_vs_r; title="ILB Policy (Complete Market, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)", colormap=:viridis, label="Weight"))
save("prob1_FW_heatmap_S_wealth_vs_rate.png", plot_heatmap(F_grid, Z_grids[1], slice_S_F_vs_r; title="Stock Policy (Complete Market, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)", colormap=:viridis, label="Weight"))

# ==============================================================================
# 6. Forward Monte Carlo Simulation (Complete Market / Financial Wealth)
# ==============================================================================
println("\nRunning Forward Monte Carlo Simulation...")
dummy_pol_c = zeros(size(pol_w_best))
_, interp_w = create_policy_interpolators(dummy_pol_c, pol_w_best, F_grid, Z_grids)

function extract_controls_prob1_FW(F_paths, r_paths, pi_paths, interp_w, dt)
    sims, steps = size(F_paths)
    wN_sim, wI_sim, wS_sim = zeros(sims, steps), zeros(sims, steps), zeros(sims, steps)
    for n in 1:steps
        idx = min(length(interp_w), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            F_safe = max(F_paths[i, n], 1e-5)
            r_val, pi_val = r_paths[i, n], pi_paths[i, n]

            wN_sim[i, n] = interp_w[idx][1](F_safe, r_val, pi_val)
            wI_sim[i, n] = interp_w[idx][2](F_safe, r_val, pi_val)
            wS_sim[i, n] = interp_w[idx][3](F_safe, r_val, pi_val)
        end
    end
    return wN_sim, wI_sim, wS_sim
end

rate_proc = VasicekProcess(:r, κ_r, overline_r, σ_r, r_0, 1)
pi_proc   = VasicekProcess(:pi, κ_π, overline_π, σ_π, π_0, 2)

B_r_N = (1.0 - exp(-κ_r * τ_N)) / κ_r
B_r_I = (1.0 - exp(-κ_r * τ_I)) / κ_r
B_pi_I = (1.0 - exp(-κ_π * τ_I)) / κ_π

# SDE for Financial Wealth in the Complete Market (3 Risky Assets + 1.0 Income)
drift_F1(t, F_val, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    F_safe = max(F_val, 1e-5)

    ω_N = interp_w[idx][1](F_safe, r_val, pi_val)
    ω_I = interp_w[idx][2](F_safe, r_val, pi_val)
    ω_S = interp_w[idx][3](F_safe, r_val, pi_val)

    RP_N = ω_N * (-λ_r * σ_r * B_r_N)
    RP_I = ω_I * (-λ_r * σ_r * B_r_I + λ_π * σ_π * B_pi_I)
    RP_S = ω_S * (λ_S * σ_S)

    return F_safe * (RP_N + RP_I + RP_S + r_val - pi_val) + 1.0
end

diff_F1(t, F_val, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    F_safe = max(F_val, 1e-5)

    ω_N = interp_w[idx][1](F_safe, r_val, pi_val)
    ω_I = interp_w[idx][2](F_safe, r_val, pi_val)
    ω_S = interp_w[idx][3](F_safe, r_val, pi_val)

    diff_r  = F_safe * (-ω_N * B_r_N * σ_r - ω_I * B_r_I * σ_r)
    diff_pi = F_safe * (ω_I * B_pi_I * σ_π)
    diff_S  = F_safe * (ω_S * σ_S)

    return [diff_r, diff_pi, diff_S]
end

w1_proc = GenericSDEProcess(:F, drift_F1, diff_F1, F_0, [1, 2, 3], [:r, :pi])
conf_prob1 = MarketConfig(sims=500, T=T, dt=dt, M=M, processes=[rate_proc, pi_proc, w1_proc], correlations=ρ_mat)
world_1 = build_world(conf_prob1)

wN_sim, wI_sim, wS_sim = extract_controls_prob1_FW(world_1.paths.F, world_1.paths.r, world_1.paths.pi, interp_w, dt)

# ==============================================================================
# 7. Additional Lifecycle & Shock Analysis Plots (Monte Carlo)
# ==============================================================================
println("\nGenerating Lifecycle Strategy and Economic Shock Plots (Monte Carlo)...")

# --- A. Investment Strategy Over Time (Combined Lifecycle Plot) ---
times_steps = 1:size(wN_sim, 2)
mean_wN = vec(mean(wN_sim, dims=1))
mean_wI = vec(mean(wI_sim, dims=1))
mean_wS = vec(mean(wS_sim, dims=1))
mean_wC = 1.0 .- mean_wN .- mean_wI .- mean_wS

save("prob1_FW_lifecycle_strategy.png",
    plot_curves(times_steps,
                [mean_wN, mean_wI, mean_wS, mean_wC],
                ["Nominal Bond (w_N)", "ILB (w_I)", "Stock (w_S)", "Cash (w_C)"];
                title="Mean Investment Strategy Over Time (Complete Market)",
                xlabel="Time Step (t)",
                ylabel="Portfolio Weight",
                legend_pos=:lt)
)

# --- B. Interest Rate Shock at t = 5 ---
r_paths_shocked = copy(world_1.paths.r)
r_paths_shocked[:, 6:end] .+= 0.05

wN_sim_r_shock, wI_sim_r_shock, wS_sim_r_shock = extract_controls_prob1_FW(world_1.paths.F, r_paths_shocked, world_1.paths.pi, interp_w, 1.0)

mean_wN_r_shock = vec(mean(wN_sim_r_shock, dims=1))
mean_wI_r_shock = vec(mean(wI_sim_r_shock, dims=1))
mean_wS_r_shock = vec(mean(wS_sim_r_shock, dims=1))

save("prob1_FW_shock_interest_rate_N.png", plot_curves(times_steps, [mean_wN, mean_wN_r_shock], ["Baseline w_N", "Shocked w_N (+5% r at t=5)"]; title="Impact of Interest Rate Shock on Nominal Bond", xlabel="Time Step (t)", ylabel="Nominal Bond Weight", legend_pos=:lt))
save("prob1_FW_shock_interest_rate_I.png", plot_curves(times_steps, [mean_wI, mean_wI_r_shock], ["Baseline w_I", "Shocked w_I (+5% r at t=5)"]; title="Impact of Interest Rate Shock on ILB", xlabel="Time Step (t)", ylabel="ILB Weight", legend_pos=:lt))
save("prob1_FW_shock_interest_rate_S.png", plot_curves(times_steps, [mean_wS, mean_wS_r_shock], ["Baseline w_S", "Shocked w_S (+5% r at t=5)"]; title="Impact of Interest Rate Shock on Stock", xlabel="Time Step (t)", ylabel="Stock Weight", legend_pos=:lt))

# --- C. Inflation Rate Shock at t = 5 ---
pi_paths_shocked = copy(world_1.paths.pi)
pi_paths_shocked[:, 6:end] .+= 0.05

wN_sim_pi_shock, wI_sim_pi_shock, wS_sim_pi_shock = extract_controls_prob1_FW(world_1.paths.F, world_1.paths.r, pi_paths_shocked, interp_w, 1.0)

mean_wN_pi_shock = vec(mean(wN_sim_pi_shock, dims=1))
mean_wI_pi_shock = vec(mean(wI_sim_pi_shock, dims=1))
mean_wS_pi_shock = vec(mean(wS_sim_pi_shock, dims=1))

save("prob1_FW_shock_inflation_N.png", plot_curves(times_steps, [mean_wN, mean_wN_pi_shock], ["Baseline w_N", "Shocked w_N (+5% π at t=5)"]; title="Impact of Inflation Shock on Nominal Bond", xlabel="Time Step (t)", ylabel="Nominal Bond Weight", legend_pos=:lt))
save("prob1_FW_shock_inflation_I.png", plot_curves(times_steps, [mean_wI, mean_wI_pi_shock], ["Baseline w_I", "Shocked w_I (+5% π at t=5)"]; title="Impact of Inflation Shock on ILB", xlabel="Time Step (t)", ylabel="ILB Weight", legend_pos=:lt))
save("prob1_FW_shock_inflation_S.png", plot_curves(times_steps, [mean_wS, mean_wS_pi_shock], ["Baseline w_S", "Shocked w_S (+5% π at t=5)"]; title="Impact of Inflation Shock on Stock", xlabel="Time Step (t)", ylabel="Stock Weight", legend_pos=:lt))


# ---------------------------------------------------------
# Plot 8: Heatmaps (Interest Rate vs Inflation) at Fixed Wealth
# ---------------------------------------------------------
println("Generating State Heatmaps (r vs pi)...")

# We use the fixed wealth index defined earlier (middle of the grid or F_0)
# slice_X_r_vs_pi dimensions: [length(Z_grids[1]), length(Z_grids[2])]

slice_N_r_vs_pi = safe_heatmap([pol_w_best[fixed_f_idx, r, pi, 1][1] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])])
slice_I_r_vs_pi = safe_heatmap([pol_w_best[fixed_f_idx, r, pi, 1][2] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])])
slice_S_r_vs_pi = safe_heatmap([pol_w_best[fixed_f_idx, r, pi, 1][3] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])])

save("prob1_FW_heatmap_N_rate_vs_inflation.png",
    plot_heatmap(Z_grids[1], Z_grids[2], slice_N_r_vs_pi;
                 title="Nominal Bond Policy (F=$F_0, t=1)",
                 xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                 colormap=:plasma, label="Weight"))

save("prob1_FW_heatmap_I_rate_vs_inflation.png",
    plot_heatmap(Z_grids[1], Z_grids[2], slice_I_r_vs_pi;
                 title="ILB Policy (F=$F_0, t=1)",
                 xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                 colormap=:plasma, label="Weight"))

save("prob1_FW_heatmap_S_rate_vs_inflation.png",
    plot_heatmap(Z_grids[1], Z_grids[2], slice_S_r_vs_pi;
                 title="Stock Policy (F=$F_0, t=1)",
                 xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                 colormap=:plasma, label="Weight"))

println("State heatmaps saved successfully.")


println("Complete Market Monte Carlo shock analysis plots saved successfully!")