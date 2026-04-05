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
# 1. Global Parameters
# ==============================================================================
M, dt, γ = 10, 1.0, 2.0
u(x) = (x^(1 - γ)) / (1 - γ)
inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# Economic Parameters
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π = 0.05, 0.02, 0.02
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N = 10.0

ρ_rπ, ρ_rS, ρ_πS = 0.5, 0.5, 0.5
ρ_mat = [1.0 ρ_rπ ρ_rS; ρ_rπ 1.0 ρ_πS; ρ_rS ρ_πS 1.0]

F_0, r_0, π_0 = 140.0, 0.02, 0.02

# ==============================================================================
# 2. Reduced Grids for Performance
# ==============================================================================
G_f = 50
F_grid = generate_log_spaced_grid(10.0, 300.0, G_f)
Z_grids = [
    generate_linear_grid(-0.02, 0.06, 10),
    generate_linear_grid(-0.02, 0.06, 10)
]

omega_space = SVector{2, Float64}[]
for w_N in range(0.0, 3.0, length=11)
    for w_S in range(0.5, 2.5, length=11)
        push!(omega_space, SVector(w_N, w_S))
    end
end
ε_nodes, W_weights = generate_gaussian_shocks(3, 5, ρ_mat)

function make_problem3_transition(κ_r, θ_r, σ_r, λ_r, τ_N, κ_π, θ_π, σ_π, λ_S, σ_S, dt)
    B_r_N = abs(κ_r) < 1e-8 ? τ_N : (1.0 - exp(-κ_r * τ_N)) / κ_r
    vol_N_r = -B_r_N * σ_r
    var_N, var_S = vol_N_r^2, σ_S^2

    return function(Z, ε)
        r_n, π_n = Z[1], Z[2]
        ε_r, ε_π, ε_S = ε[1], ε[2], ε[3]

        r_next = clamp(r_n + κ_r * (θ_r - r_n) * dt + σ_r * sqrt(dt) * ε_r, -0.02, 0.06)
        π_next = clamp(π_n + κ_π * (θ_π - π_n) * dt + σ_π * sqrt(dt) * ε_π, -0.06, 0.10)
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
    return max(F * (dot(ω, R_e) + R_base) + 1.0 * dt, 1e-2)
end

crra_ex = make_crra_extrapolator(F_grid[1], F_grid[end], γ)

# ==============================================================================
# 3. Solvers & Benchmarking
# ==============================================================================
solvers = [
    # ("BruteForce", BruteForceSolver()), # Uncomment to test baseline
    ("Zooming", ZoomingSolver(iterations=3, points_per_dim=20, zoom_range_factor=1.2)),
    # ("Optim", OptimSolver(use_gradients=true, coarse_warm_start_n=5, optim_options=Optim.Options(
    #     iterations = 50 # Prevent it from looping forever if it gets stuck
    # )))
]

results = []
pol_w_best = nothing
V_best = nothing
best_CE = -Inf # Track the highest CE found to use for plotting

println("\nWarming up (pre-compiling) solvers...")
for (name, solver) in solvers
    solve_dynamic_program(solver, F_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob3, 1, u, identity, problem3_budget_constraint, crra_ex)
end

println("\nRunning Benchmarks (M = 10)...")
for (name, solver) in solvers
    println("  Running $name solver...")

    local V, pol_w
    time_taken = @elapsed begin
        V, pol_w = solve_dynamic_program(solver, F_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob3, M, u, identity, problem3_budget_constraint, crra_ex)
    end

    # Extract Certainty Equivalent
    V_interp = linear_interpolation((F_grid, Z_grids[1], Z_grids[2]), V[:, :, :, 1], extrapolation_bc=Line())
    CE_num = calculate_certainty_equivalent(V_interp(F_0, r_0, π_0), inv_u)

    # Dynamically save the best policy
    global best_CE, pol_w_best, V_best
    if CE_num > best_CE
        best_CE = CE_num
        pol_w_best = pol_w
        V_best = V
    end

    push!(results, (name, time_taken, CE_num))
end

println("\n==================================================")
println("PERFORMANCE & ACCURACY REPORT (Problem 3)")
println("==================================================")
for (name, time_taken, CE_num) in results
    println(rpad(name, 12), " | Time: ", rpad("$(round(time_taken, digits=2))s", 8),
            " | CE (W_0): ", rpad(round(CE_num, digits=4), 8))
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
conf_prob3 = MarketConfig(sims=500, T=10.0, dt=1.0, M=10, processes=[rate_proc, pi_proc, w3_proc], correlations=ρ_mat)
world_3 = build_world(conf_prob3)

wN_sim, wS_sim = extract_controls_prob3(world_3.paths.F, world_3.paths.r, world_3.paths.pi, interp_w, 1.0)

# ==============================================================================
# 5. Full Plotting Suite
# ==============================================================================
println("Generating and saving plots...")
fixed_f_idx = div(G_f, 2)
fixed_F_val = round(F_grid[fixed_f_idx], digits=2)
fixed_r_idx = 3; fixed_pi_idx = 3

# Plot 1 & 2: Value & CE
save("prob3_adv_value_function.png", plot_curves(F_grid, [V_best[:, fixed_r_idx, fixed_pi_idx, 1], V_best[:, fixed_r_idx, fixed_pi_idx, 5], V_best[:, fixed_r_idx, fixed_pi_idx, 10]], ["t = 1", "t = 5", "t = 10"]; title="Expected Utility V(F) (r=0.02, π=0.02)", xlabel="Financial Wealth (F)", ylabel="Utility", legend_pos=:rb))
ce_over_time = [calculate_certainty_equivalent(V_best[fixed_f_idx, fixed_r_idx, fixed_pi_idx, t], inv_u) for t in 1:M]
save("prob3_adv_ce_progression.png", plot_curves(1:M, [ce_over_time], ["CE Financial Wealth"]; title="CE Progression (F=$fixed_F_val, r=0.02, π=0.02)", xlabel="Time Step (t)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rt))

# Plot 3-4: Mean Strategy Allocations
save("prob3_adv_mean_nominal_bond.png", plot_mean_with_bounds(wN_sim; title="Mean Nominal Bond Allocation", ylabel="Weight", color=:blue))
save("prob3_adv_mean_stock.png", plot_mean_with_bounds(wS_sim; title="Mean Stock Allocation", ylabel="Weight", color=:green))

# Plot 5-6: Heatmaps (Interest Rate vs Inflation)
slice_N_r_vs_pi = [pol_w_best[fixed_f_idx, r, pi, 1][1] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]
slice_S_r_vs_pi = [pol_w_best[fixed_f_idx, r, pi, 1][2] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]
save("prob3_adv_heatmap_N_rate_vs_inflation.png", plot_heatmap(Z_grids[1], Z_grids[2], slice_N_r_vs_pi; title="Nominal Bond Policy (F=$fixed_F_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("prob3_adv_heatmap_S_rate_vs_inflation.png", plot_heatmap(Z_grids[1], Z_grids[2], slice_S_r_vs_pi; title="Stock Policy (F=$fixed_F_val, t=1)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

# Plot 7-8: Heatmaps (Wealth vs Interest Rate)
slice_N_F_vs_r = [pol_w_best[f, r, fixed_pi_idx, 1][1] for f in 1:length(F_grid), r in 1:length(Z_grids[1])]
slice_S_F_vs_r = [pol_w_best[f, r, fixed_pi_idx, 1][2] for f in 1:length(F_grid), r in 1:length(Z_grids[1])]
save("prob3_adv_heatmap_N_wealth_vs_rate.png", plot_heatmap(F_grid, Z_grids[1], slice_N_F_vs_r; title="Nominal Bond Policy (π=0.02, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)", colormap=:viridis, label="Weight"))
save("prob3_adv_heatmap_S_wealth_vs_rate.png", plot_heatmap(F_grid, Z_grids[1], slice_S_F_vs_r; title="Stock Policy (π=0.02, t=1)", xlabel="Financial Wealth (F)", ylabel="Interest Rate (r)", colormap=:viridis, label="Weight"))

# Plot 9: Wealth vs Human Capital Evolution
times_seq = 1:M
H_mean = [(M - t + 1) * dt for t in times_seq]
F_mean = vec(mean(world_3.paths.F, dims=1))[1:M]
save("prob3_adv_wealth_composition.png", plot_curves(times_seq, [F_mean, H_mean], ["Financial Wealth (F)", "Human Capital (H)"]; title="Wealth Composition Over Time", xlabel="Time (Steps)", ylabel="Value", legend_pos=:rc))

println("All Problem 3 plots and evaluations complete!")