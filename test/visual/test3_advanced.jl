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
println("Advanced Solvers: Problem 3 (Incomplete Market)")
println("==================================================")

# ==============================================================================
# 1. Global Parameters & Utilities
# ==============================================================================
const M = 20
const dt = 0.5
const γ = 5.0

# Use a factory function to strictly type the utilities and prevent boxing/allocations
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
κ_π, overline_π, σ_π = 0.05, 0.02, 0.02
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N = 10.0

ρ_rπ, ρ_rS, ρ_πS = 0.5, -0.2, -0.2
ρ_mat = [1.0 ρ_rπ ρ_rS; ρ_rπ 1.0 ρ_πS; ρ_rS ρ_πS 1.0]

F_0, r_0, π_0 = 140.0, 0.02, 0.02

# ==============================================================================
# 2. Reduced Grids for Performance
# ==============================================================================
G_f = 50
F_grid = generate_log_spaced_grid(10.0, 300.0, G_f)
Z_grids = [
    generate_linear_grid(-0.04, 0.08, 15),
    generate_linear_grid(-0.04, 0.08, 15)
]

omega_space = SVector{2, Float64}[]
for w_N in range(-1.0, 3.0, length=11)
    for w_S in range(-1.0, 2.5, length=11)
        push!(omega_space, SVector(w_N, w_S))
    end
end
ε_nodes, W_weights = generate_gaussian_shocks(3, 6, ρ_mat)

function make_problem3_transition(κ_r, θ_r, σ_r, λ_r, τ_N, κ_π, θ_π, σ_π, λ_S, σ_S, dt)
    B_r_N = abs(κ_r) < 1e-8 ? τ_N : (1.0 - exp(-κ_r * τ_N)) / κ_r
    vol_N_r = -B_r_N * σ_r
    var_N, var_S = vol_N_r^2, σ_S^2

    return function(Z, ε)
        r_n, π_n = Z[1], Z[2]
        ε_r, ε_π, ε_S = ε[1], ε[2], ε[3]

        r_next = clamp(r_n + κ_r * (θ_r - r_n) * dt + σ_r * sqrt(dt) * ε_r, -0.04, 0.08)
        π_next = clamp(π_n + κ_π * (θ_π - π_n) * dt + σ_π * sqrt(dt) * ε_π, -0.04, 0.08)
        Z_next = SVector(r_next, π_next)

        Rf_nom = exp(r_n * dt)
        R_N = exp((r_n - λ_r * σ_r * B_r_N - 0.5 * var_N) * dt + vol_N_r * sqrt(dt) * ε_r)
        R_S = exp((r_n + λ_S * σ_S - 0.5 * var_S) * dt + σ_S * sqrt(dt) * ε_S)

        Re = SVector(R_N - Rf_nom, R_S - Rf_nom)
        R_base_real = exp((r_n - π_n) * dt)
        return Z_next, Re, R_base_real
    end
end

transition_prob3 = make_problem3_transition(κ_r, overline_r, σ_r, λ_r, τ_N, κ_π, overline_π, σ_π, λ_S, σ_S, dt)

# === THE FIX: Changed 1e-10 to 1e-2 to prevent NaN Gradients in Optim! ===
function problem3_budget_constraint(F, c, ω, R_e, R_base)
    return max(F * (dot(ω, R_e) + R_base) + 1.0 * dt, 1e-10)
end

# Use the new CE extrapolator (no γ needed!)
ce_ex = make_ce_crra_extrapolator(F_grid[1], F_grid[end])

# ==============================================================================
# 3. Solvers & Benchmarking
# ==============================================================================
solvers = [
    # ("BruteForce", BruteForceSolver()), # Uncomment to test baseline
    # ("Zooming", ZoomingSolver(iterations=4, points_per_dim=10, zoom_range_factor=1.1)),
    ("Optic", OptimSolver(use_gradients=true, coarse_warm_start_n=5, optim_options=Optim.Options(
        iterations = 100 # Prevent it from looping forever if it gets stuck
    )))
]

results = []
pol_w_best = nothing
CE_best = nothing
best_CE_val = -Inf # Track the highest CE found to use for plotting

println("\nWarming up (pre-compiling) solvers...")
for (name, solver) in solvers
    solve_dynamic_program(solver, F_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob3, 1, u, inv_u, identity, problem3_budget_constraint, ce_ex)
end

println("\nRunning Benchmarks (M = 10)...")
for (name, solver) in solvers
    println("  Running $name solver...")

    local CE_grid, pol_w
    time_taken = @elapsed begin
        CE_grid, pol_w = solve_dynamic_program(solver, F_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob3, M, u, inv_u, identity, problem3_budget_constraint, ce_ex)
    end

    # Extract Certainty Equivalent directly (no inverse utility calculation required anymore)
    CE_interp = linear_interpolation((F_grid, Z_grids[1], Z_grids[2]), CE_grid[:, :, :, 1], extrapolation_bc=Line())
    CE_num = CE_interp(F_0, r_0, π_0)

    # Dynamically save the best policy
    global best_CE_val, pol_w_best, CE_best
    if CE_num > best_CE_val
        best_CE_val = CE_num
        pol_w_best = pol_w
        CE_best = CE_grid
    end

    push!(results, (name, time_taken, CE_num))
end

# ==============================================================================
# 3b. Evaluate Certainty Equivalent Ratios for Multiple Wealth Levels (Problem 3)
# ==============================================================================
println("\n==================================================")
println("PROBLEM 3: CE ANALYSIS ACROSS WEALTH LEVELS")
println("==================================================")

# We need the exact Human Capital from Problem 1 to use as our "shadow benchmark".
H_0_complete = 9.8839
T_years = M * dt

# Create an interpolator from the best solver's CE grid at t=1
CE_interp = linear_interpolation((F_grid, Z_grids[1], Z_grids[2]), CE_best[:, :, :, 1], extrapolation_bc=Line())

println("\n--- Wealth Variation & % CE Analysis (Incomplete Market) ---")
println(rpad("F_0", 8), rpad("H_0/F_0", 10), rpad("W_0 (Base)", 12), rpad("CE (Abs)", 12), rpad("Total % CE", 12), rpad("Real CE Gain", 15), "Nominal CE Gain")
println("-"^85)

F_test_values = [50.0, 100.0, 140.0, 300.0]

for F_val in F_test_values
    # 1. Calculate the shadow benchmark Total Wealth
    W_base = F_val + H_0_complete
    HC_to_F_ratio = H_0_complete / F_val

    # 2. Extract Absolute Certainty Equivalent directly from the grid!
    CE_val = CE_interp(F_val, r_0, π_0)

    # 3. Calculate Metrics relative to W_base
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

# Print the performance metrics below
println("\n--- Solver Computation Times ---")
for (name, time_taken, CE_val) in results
    println(rpad(name, 12), " | Time: ", rpad("$(round(time_taken, digits=2))s", 8))
end
println("==================================================")

# ==============================================================================
# 4. Forward Monte Carlo Simulation
# ==============================================================================
println("\nRunning Forward Monte Carlo Simulation...")
dummy_pol_c = zeros(size(pol_w_best))
_, interp_w = create_policy_interpolators(dummy_pol_c, pol_w_best, F_grid, Z_grids)

function extract_controls_prob3(F_paths, r_paths, pi_paths, interp_w, dt)
    sims, steps = size(F_paths)
    wN_sim, wS_sim = zeros(sims, steps), zeros(sims, steps)
    for n in 1:steps
        idx = min(length(interp_w), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            wN_sim[i, n] = interp_w[idx][1](max(F_paths[i, n], 1e-5), r_paths[i, n], pi_paths[i, n])
            wS_sim[i, n] = interp_w[idx][2](max(F_paths[i, n], 1e-5), r_paths[i, n], pi_paths[i, n])
        end
    end
    return wN_sim, wS_sim
end

rate_proc = VasicekProcess(:r, κ_r, overline_r, σ_r, r_0, 1)
pi_proc   = VasicekProcess(:pi, κ_π, overline_π, σ_π, π_0, 2)
B_r_N = (1.0 - exp(-κ_r * τ_N)) / κ_r

drift_F3(t, F_val, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    F_safe = max(F_val, 1e-5)
    ω_N, ω_S = interp_w[idx][1](F_safe, r_val, pi_val), interp_w[idx][2](F_safe, r_val, pi_val)
    return F_safe * (ω_N * (-λ_r * σ_r * B_r_N) + ω_S * (λ_S * σ_S) + r_val - pi_val) + 1.0
end

diff_F3(t, F_val, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    F_safe = max(F_val, 1e-5)
    ω_N, ω_S = interp_w[idx][1](F_safe, r_val, pi_val), interp_w[idx][2](F_safe, r_val, pi_val)
    return [-F_safe * ω_N * B_r_N * σ_r, 0.0, F_safe * ω_S * σ_S]
end

w3_proc = GenericSDEProcess(:F, drift_F3, diff_F3, F_0, [1, 2, 3], [:r, :pi])
conf_prob3 = MarketConfig(sims=500, T=M * dt, dt=dt, M=M, processes=[rate_proc, pi_proc, w3_proc], correlations=ρ_mat)
world_3 = build_world(conf_prob3)

wN_sim, wS_sim = extract_controls_prob3(world_3.paths.F, world_3.paths.r, world_3.paths.pi, interp_w, 1.0)

# ==============================================================================
# 5. Full Plotting Suite
# ==============================================================================
println("Generating and saving plots...")
fixed_f_idx = div(G_f, 2)
fixed_F_val = round(F_grid[fixed_f_idx], digits=2)
fixed_r_idx = 3; fixed_pi_idx = 3

# Plot 1 & 2: Value (now CE) & CE Progression
save("prob3_adv_value_function.png", plot_curves(F_grid, [CE_best[:, fixed_r_idx, fixed_pi_idx, 1], CE_best[:, fixed_r_idx, fixed_pi_idx, 5], CE_best[:, fixed_r_idx, fixed_pi_idx, 10]], ["t = 1", "t = 5", "t = 10"]; title="Certainty Equivalent CE(F) (r=0.02, π=0.02)", xlabel="Financial Wealth (F)", ylabel="CE (Dollars)", legend_pos=:rb))
ce_over_time = [CE_best[fixed_f_idx, fixed_r_idx, fixed_pi_idx, t] for t in 1:M]
save("prob3_adv_ce_progression.png", plot_curves(1:M, [ce_over_time], ["CE Financial Wealth"]; title="CE Progression (F=$fixed_F_val, r=0.02, π=0.02)", xlabel="Time Step (t)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rt))

# Plot 3-4: Mean Strategy Allocations
save("prob3_adv_mean_nominal_bond.png", plot_mean_with_bounds(wN_sim; title="Mean Nominal Bond Allocation", ylabel="Weight", color=:blue))
save("prob3_adv_mean_stock.png", plot_mean_with_bounds(wS_sim; title="Mean Stock Allocation", ylabel="Weight", color=:green))

# --- SAFEGUARD FUNCTION FOR CAIROMAKIE ---
function safe_heatmap(data)
    safe_data = copy(data)
    if maximum(safe_data) ≈ minimum(safe_data)
        safe_data[1, 1] += 1e-5
        safe_data[end, end] -= 1e-5
    end
    return safe_data
end
# -----------------------------------------

# Plot 5-6: Heatmaps (Interest Rate vs Inflation)
slice_N_r_vs_pi = safe_heatmap([pol_w_best[fixed_f_idx, r, pi, 1][1] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])])
slice_S_r_vs_pi = safe_heatmap([pol_w_best[fixed_f_idx, r, pi, 1][2] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])])

save("prob3_adv_heatmap_N_rate_vs_inflation.png", plot_heatmap(Z_grids[1], Z_grids[2], slice_N_r_vs_pi; title="Nominal Bond Policy (F=$fixed_F_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("prob3_adv_heatmap_S_rate_vs_inflation.png", plot_heatmap(Z_grids[1], Z_grids[2], slice_S_r_vs_pi; title="Stock Policy (F=$fixed_F_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

# Plot 7-8: Heatmaps (Wealth vs Interest Rate)
slice_N_F_vs_r = safe_heatmap([pol_w_best[f, r, fixed_pi_idx, 1][1] for f in 1:length(F_grid), r in 1:length(Z_grids[1])])
slice_S_F_vs_r = safe_heatmap([pol_w_best[f, r, fixed_pi_idx, 1][2] for f in 1:length(F_grid), r in 1:length(Z_grids[1])])

save("prob3_adv_heatmap_N_wealth_vs_rate.png", plot_heatmap(F_grid, Z_grids[1], slice_N_F_vs_r; title="Nominal Bond Policy (π=0.02, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)", colormap=:viridis, label="Weight"))
save("prob3_adv_heatmap_S_wealth_vs_rate.png", plot_heatmap(F_grid, Z_grids[1], slice_S_F_vs_r; title="Stock Policy (π=0.02, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)", colormap=:viridis, label="Weight"))

# Plot 9: Wealth vs Human Capital Evolution
times_seq = 1:M
H_mean = [(M - t + 1) * dt for t in times_seq]
F_mean = vec(mean(world_3.paths.F, dims=1))[1:M]
save("prob3_adv_wealth_composition.png", plot_curves(times_seq, [F_mean, H_mean], ["Financial Wealth (F)", "Human Capital (H)"]; title="Wealth Composition Over Time", xlabel="Time (Steps)", ylabel="Value", legend_pos=:rc))

println("All Problem 3 plots and evaluations complete!")
# ==============================================================================
# 6. Additional Lifecycle & Shock Analysis Plots (Monte Carlo)
# ==============================================================================
println("\nGenerating Lifecycle Strategy and Economic Shock Plots (Monte Carlo)...")

# --- A. Investment Strategy Over Time (Combined Lifecycle Plot) ---
# Calculate the mean allocation across all 500 simulations at each time step
times_steps = 1:size(wN_sim, 2)
mean_wN = vec(mean(wN_sim, dims=1))
mean_wS = vec(mean(wS_sim, dims=1))
# Assuming the rest of the wealth is held in the risk-free cash account (w_C = 1 - w_N - w_S)
mean_wC = 1.0 .- mean_wN .- mean_wS

save("prob3_adv_lifecycle_strategy.png",
    plot_curves(times_steps,
                [mean_wN, mean_wS, mean_wC],
                ["Nominal Bond (w_N)", "Stock (w_S)", "Cash (w_C)"];
                title="Mean Investment Strategy Over Time (Prob 3)",
                xlabel="Time Step (t)",
                ylabel="Portfolio Weight",
                legend_pos=:lt)
)

# --- B. Interest Rate Shock at t = 5 ---
# Create a copy of the simulated paths and apply a +5% (0.05) shock at t=5 (index 6) onwards
r_paths_shocked = copy(world_3.paths.r)
r_paths_shocked[:, 6:end] .+= 0.05

# Re-extract the controls using the shocked interest rate paths to isolate the policy response
wN_sim_r_shock, wS_sim_r_shock = extract_controls_prob3(world_3.paths.F, r_paths_shocked, world_3.paths.pi, interp_w, 1.0)

mean_wN_r_shock = vec(mean(wN_sim_r_shock, dims=1))
mean_wS_r_shock = vec(mean(wS_sim_r_shock, dims=1))

save("prob3_adv_shock_interest_rate_N.png",
    plot_curves(times_steps,
                [mean_wN, mean_wN_r_shock],
                ["Baseline w_N", "Shocked w_N (+5% r at t=5)"];
                title="Impact of Interest Rate Shock on Nominal Bond Allocation",
                xlabel="Time Step (t)",
                ylabel="Nominal Bond Weight",
                legend_pos=:lt)
)

save("prob3_adv_shock_interest_rate_S.png",
    plot_curves(times_steps,
                [mean_wS, mean_wS_r_shock],
                ["Baseline w_S", "Shocked w_S (+5% r at t=5)"];
                title="Impact of Interest Rate Shock on Stock Allocation",
                xlabel="Time Step (t)",
                ylabel="Stock Weight",
                legend_pos=:lt)
)

# --- C. Inflation Rate Shock at t = 5 ---
# Create a copy of the simulated paths and apply a +5% (0.05) shock at t=5 (index 6) onwards
pi_paths_shocked = copy(world_3.paths.pi)
pi_paths_shocked[:, 6:end] .+= 0.05

# Re-extract the controls using the shocked inflation paths
wN_sim_pi_shock, wS_sim_pi_shock = extract_controls_prob3(world_3.paths.F, world_3.paths.r, pi_paths_shocked, interp_w, 1.0)

mean_wN_pi_shock = vec(mean(wN_sim_pi_shock, dims=1))
mean_wS_pi_shock = vec(mean(wS_sim_pi_shock, dims=1))

save("prob3_adv_shock_inflation_N.png",
    plot_curves(times_steps,
                [mean_wN, mean_wN_pi_shock],
                ["Baseline w_N", "Shocked w_N (+5% π at t=5)"];
                title="Impact of Inflation Shock on Nominal Bond Allocation",
                xlabel="Time Step (t)",
                ylabel="Nominal Bond Weight",
                legend_pos=:lt)
)

save("prob3_adv_shock_inflation_S.png",
    plot_curves(times_steps,
                [mean_wS, mean_wS_pi_shock],
                ["Baseline w_S", "Shocked w_S (+5% π at t=5)"];
                title="Impact of Inflation Shock on Stock Allocation",
                xlabel="Time Step (t)",
                ylabel="Stock Weight",
                legend_pos=:lt)
)

println("Problem 3 Monte Carlo shock analysis plots saved successfully!")