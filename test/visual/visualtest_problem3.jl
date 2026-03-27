using BruteForceAllocationSolver
using FinancialMarketSimulation
using FastGaussQuadrature
using CairoMakie
using LinearAlgebra
using Statistics

# ==============================================================================
# HELPER: Extract Controls from Simulated Paths (Terminal Wealth - No Consumption)
# ==============================================================================
function extract_controls_3d_terminal(W_paths, r_paths, pi_paths, interp_w, dt)
    sims, steps = size(W_paths)
    wN_sim = zeros(sims, steps)
    wS_sim = zeros(sims, steps)

    for n in 1:steps
        idx = min(length(interp_w), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            W = max(W_paths[i, n], 1e-5) # Prevent bounds errors
            r = r_paths[i, n]
            π_val = pi_paths[i, n]

            wN_sim[i, n] = interp_w[idx][1](W, r, π_val)
            wS_sim[i, n] = interp_w[idx][2](W, r, π_val)
        end
    end
    return wN_sim, wS_sim
end

# ==============================================================================
# 1. DP Setup & Solve (Problem 3: Incomplete Market - Terminal Wealth)
# ==============================================================================
println("Solving Problem 3 DP (Terminal Wealth, No Consumption)...")

M, dt, γ = 10, 1.0, 5.0
u(x) = (x^(1 - γ)) / (1 - γ)
inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ)) # Inverse for Certainty Equivalent

# Economic Parameters
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π = 0.05, 0.02, 0.02
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.1
τ = 10.0 # Rolling bond maturity

# Correlations
ρ_rπ, ρ_rS, ρ_πS = 0.5, 0.5, 0.5
ρ_mat = [
    1.0   ρ_rπ  ρ_rS;
    ρ_rπ  1.0   ρ_πS;
    ρ_rS  ρ_πS  1.0
]

# Grids (NO consumption grid!)
G_w = 400
W_grid = generate_log_spaced_grid(0.5, 100.0, G_w)
Z_grids = [
    generate_linear_grid(0.0, 0.1, 10),  # r_grid (Center index 3 is 0.02)
    generate_linear_grid(0.0, 0.1, 10)   # π_grid (Center index 3 is 0.02)
]

# Expand the boundaries and increase the density
omega_space = Vector{Float64}[]
for w_N in range(-3.0, 5.0, length=41)   # Nominal bond from -300% to 500%
    for w_S in range(-1.0, 2.5, length=41) # Stock from 0% to 150%
        push!(omega_space, [w_N, w_S])
    end
end

# 3D Quadrature Nodes
ε_nodes, W_weights = generate_gaussian_shocks(3, 3, ρ_mat)

# Custom Transition for Problem 3
function make_problem3_transition(κ_r, θ_r, σ_r, λ_r, τ, κ_π, θ_π, σ_π, λ_S, σ_S, dt)
    B_r = abs(κ_r) < 1e-8 ? τ : (1.0 - exp(-κ_r * τ)) / κ_r
    bond_vol = -B_r * σ_r

    return function(Z::Vector{Float64}, ε::Vector{Float64})
        r_n, π_n = Z[1], Z[2]
        ε_r, ε_π, ε_S = ε[1], ε[2], ε[3]

        r_next = r_n + κ_r * (θ_r - r_n) * dt + σ_r * sqrt(dt) * ε_r
        π_next = π_n + κ_π * (θ_π - π_n) * dt + σ_π * sqrt(dt) * ε_π
        Z_next = [r_next, π_next]

        Rf_nom = exp(r_n * dt)
        R_bond_nom = exp((r_n - λ_r * B_r * σ_r - 0.5 * bond_vol^2) * dt + bond_vol * sqrt(dt) * ε_r)
        R_S_nom = exp((r_n + λ_S * σ_S - 0.5 * σ_S^2) * dt + σ_S * sqrt(dt) * ε_S)

        Re = [R_bond_nom - Rf_nom, R_S_nom - Rf_nom]
        R_base_real = exp((r_n - π_n) * dt)

        return Z_next, Re, R_base_real
    end
end

transition_prob3 = make_problem3_transition(κ_r, overline_r, σ_r, λ_r, τ, κ_π, overline_π, σ_π, λ_S, σ_S, dt)

# Custom Budget Constraint (Terminal Wealth form -> c is ignored/dropped entirely)
function problem3_budget_constraint_terminal(W, c, ω, R_e, R_base)
    income_real = 1.0 * dt
    # Notice: (1.0 - c) is removed since consumption is zero
    return W * (dot(ω, R_e) + R_base) + income_real
end

crra_ex = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

# Call the overloaded Terminal Wealth solver (no c_grid, no β, identity for wealth space)
V, pol_w = solve_dynamic_program(
    W_grid, Z_grids, omega_space,
    ε_nodes, W_weights, transition_prob3,
    M, u, identity,
    problem3_budget_constraint_terminal, crra_ex
)

# Use a dummy consumption policy array to leverage your existing interpolator factory
dummy_pol_c = zeros(size(pol_w))
_, interp_w = create_policy_interpolators(dummy_pol_c, pol_w, W_grid, Z_grids)

# ==============================================================================
# 2. Forward Monte Carlo Simulation
# ==============================================================================
println("Running Forward Monte Carlo Simulation...")

rate_proc = VasicekProcess(:r, κ_r, overline_r, σ_r, 0.02, 1)
pi_proc   = VasicekProcess(:pi, κ_π, overline_π, σ_π, 0.02, 2)

B_r_val = (1.0 - exp(-κ_r * τ)) / κ_r

# SDE for Real Wealth
drift_W3(t, W, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    W_safe = max(W, 1e-5)
    ω_N = interp_w[idx][1](W_safe, r_val, pi_val)
    ω_S = interp_w[idx][2](W_safe, r_val, pi_val)

    # Real wealth drift with additive income (NO consumption subtracted)
    return W_safe * (ω_N * (-λ_r * σ_r * B_r_val) + ω_S * (λ_S * σ_S) + r_val - pi_val) + 1.0
end

diff_W3(t, W, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    W_safe = max(W, 1e-5)
    ω_N = interp_w[idx][1](W_safe, r_val, pi_val)
    ω_S = interp_w[idx][2](W_safe, r_val, pi_val)

    # [Diff_r, Diff_π, Diff_S]
    return [-W_safe * ω_N * B_r_val * σ_r, 0.0, W_safe * ω_S * σ_S]
end

w3_proc = GenericSDEProcess(:W, drift_W3, diff_W3, 10.0, [1, 2, 3], [:r, :pi])

conf_prob3 = MarketConfig(
    sims=500, T=10.0, dt=1.0, M=10,
    processes=[rate_proc, pi_proc, w3_proc],
    correlations=ρ_mat
)

world = build_world(conf_prob3)

# Extract simulated controls
wN_sim, wS_sim = extract_controls_3d_terminal(world.paths.W, world.paths.r, world.paths.pi, interp_w, 1.0)

# ==============================================================================
# 3. Generating Plots using `plotting.jl`
# ==============================================================================
println("Generating and saving plots...")

# ---------------------------------------------------------
# Plot 1: Agnostic Curves: Expected Utility vs Wealth
# ---------------------------------------------------------
y_data_V = [
    V[:, 3, 3, 1],  # t=1
    V[:, 3, 3, 5],  # t=5
    V[:, 3, 3, 10]  # t=10
]
labels_time = ["t = 1", "t = 5", "t = 10"]

fig_v = plot_curves(W_grid, y_data_V, labels_time;
                    title="Problem 3: Expected Utility V(W) (Terminal)",
                    xlabel="Real Wealth (W)", ylabel="Expected Utility", legend_pos=:rb)
save("prob3_term_value_function.png", fig_v)


# ---------------------------------------------------------
# Plot 2: Agnostic Curves: Certainty Equivalent vs Wealth
# ---------------------------------------------------------
y_data_CE = [
    calculate_certainty_equivalent.(V[:, 3, 3, 1], inv_u),
    calculate_certainty_equivalent.(V[:, 3, 3, 5], inv_u),
    calculate_certainty_equivalent.(V[:, 3, 3, 10], inv_u)
]

fig_ce_wealth = plot_curves(W_grid, y_data_CE, labels_time;
                            title="Problem 3: Certainty Equivalent Terminal Wealth",
                            xlabel="Current Real Wealth (W)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rb)
save("prob3_term_ce_vs_wealth.png", fig_ce_wealth)


# ---------------------------------------------------------
# Plot 3: Agnostic Curves: Certainty Equivalent Progression Over Time
# ---------------------------------------------------------
fixed_w_idx = 200 # Middle of grid
time_axis = 1:M
ce_over_time = [calculate_certainty_equivalent(V[fixed_w_idx, 3, 3, t], inv_u) for t in time_axis]

fig_ce_time = plot_curves(time_axis, [ce_over_time], ["CE Terminal Wealth"];
                          title="Certainty Equivalent Progression Over Time (Fixed State)",
                          xlabel="Time Step (t)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rt)
save("prob3_term_ce_progression.png", fig_ce_time)


# ---------------------------------------------------------
# Plot 4: Agnostic Heatmap: Wealth vs Interest Rate (Nominal Bond)
# ---------------------------------------------------------
slice_W_vs_r = [pol_w[w, r, 3, 1][1] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]

fig_heat_N1 = plot_heatmap(W_grid, Z_grids[1], slice_W_vs_r;
                           title="Nominal Bond Policy (t=1, π=0.02)",
                           xlabel="Real Wealth (W)", ylabel="Interest Rate (r)",
                           colormap=:viridis, label="Nominal Bond Weight")
save("prob3_term_heatmap_wealth_vs_rate.png", fig_heat_N1)


# ---------------------------------------------------------
# Plot 5: Agnostic Heatmap: Interest Rate vs Inflation (Nominal Bond)
# ---------------------------------------------------------
slice_r_vs_pi = [pol_w[fixed_w_idx, r, pi, 1][1] for r in 1:length(Z_grids[1]), pi in 1:length(Z_grids[2])]

fig_heat_N2 = plot_heatmap(Z_grids[1], Z_grids[2], slice_r_vs_pi;
                           title="Nominal Bond Policy (t=1, Middle Wealth)",
                           xlabel="Interest Rate (r)", ylabel="Inflation (π)",
                           colormap=:plasma, label="Nominal Bond Weight")
save("prob3_term_heatmap_rate_vs_inflation.png", fig_heat_N2)


# ---------------------------------------------------------
# Plot 6: Agnostic Heatmap: Wealth vs Interest Rate (Stock)
# ---------------------------------------------------------
slice_S_W_vs_r = [pol_w[w, r, 3, 1][2] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]

fig_heat_S = plot_heatmap(W_grid, Z_grids[1], slice_S_W_vs_r;
                          title="Stock Policy (t=1, π=0.02)",
                          xlabel="Real Wealth (W)", ylabel="Interest Rate (r)",
                          colormap=:plasma, label="Stock Weight")
save("prob3_term_heatmap_stock.png", fig_heat_S)


# ---------------------------------------------------------
# Plot 7-9: Path Overlays
# ---------------------------------------------------------
fig_paths_W = plot_paths_overlay(Matrix(world.paths.W); title="Simulated Real Wealth Paths", xlabel="Time (Years)", ylabel="Real Wealth", line_color=:dodgerblue)
save("prob3_term_sim_wealth_paths.png", fig_paths_W)

fig_paths_r = plot_paths_overlay(Matrix(world.paths.r); title="Simulated Interest Rate Paths", line_color=:firebrick)
save("prob3_term_sim_rate_paths.png", fig_paths_r)

fig_paths_pi = plot_paths_overlay(Matrix(world.paths.pi); title="Simulated Inflation Paths", line_color=:darkorange)
save("prob3_term_sim_inflation_paths.png", fig_paths_pi)


# ---------------------------------------------------------
# Plot 10-11: Mean Strategy Allocations
# ---------------------------------------------------------
fig_mean_wN = plot_mean_with_bounds(wN_sim; title="Mean Nominal Bond Allocation", ylabel="Portfolio Weight", color=:blue)
save("prob3_term_mean_nominal_bond.png", fig_mean_wN)

fig_mean_wS = plot_mean_with_bounds(wS_sim; title="Mean Stock Allocation", ylabel="Portfolio Weight", color=:green)
save("prob3_term_mean_stock.png", fig_mean_wS)


# ---------------------------------------------------------
# Plot 12: Deterministic Target-Date Glidepath
# ---------------------------------------------------------
times_seq = 1:M
glide_wN = [interp_w[t][1](50.0, 0.02, 0.02) for t in times_seq]
glide_wS = [interp_w[t][2](50.0, 0.02, 0.02) for t in times_seq]

fig_glide = plot_curves(times_seq, [glide_wN, glide_wS], ["Nominal Bond", "Stock"];
                        title="Deterministic Target-Date Glidepath",
                        xlabel="Time (Steps)", ylabel="Portfolio Weight", legend_pos=:rt)
save("prob3_term_deterministic_glidepath.png", fig_glide)


# ---------------------------------------------------------
# Plot 13: Wealth vs Human Capital Evolution
# ---------------------------------------------------------
# Expected PV of future real income discounted at real rate (approx 0.0 since r_bar = pi_bar = 0.02)
H_mean = [(M - t + 1) * dt for t in times_seq]
W_mean = vec(mean(world.paths.W, dims=1))[1:M]

fig_wealth_comp = plot_curves(times_seq, [W_mean, H_mean], ["Mean Financial Wealth (W)", "Human Capital (H)"];
                              title="Wealth Composition Over Time",
                              xlabel="Time (Steps)", ylabel="Value", legend_pos=:rc)
save("prob3_term_wealth_composition.png", fig_wealth_comp)

println("All Terminal Wealth Problem 3 solutions, simulations, and plots generated successfully!")