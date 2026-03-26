using BruteForceAllocationSolver
using FinancialMarketSimulation
using FastGaussQuadrature
using CairoMakie
using LinearAlgebra
using Statistics

# ==============================================================================
# HELPER: Extract Controls from Simulated Paths (3D State Space)
# ==============================================================================
function extract_controls_3d(W_paths, r_paths, pi_paths, interp_c, interp_w, dt)
    sims, steps = size(W_paths)
    c_sim = zeros(sims, steps)
    wN_sim = zeros(sims, steps)
    wS_sim = zeros(sims, steps)

    for n in 1:steps
        idx = min(length(interp_c), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            W = max(W_paths[i, n], 1e-5) # Prevent bounds errors
            r = r_paths[i, n]
            π_val = pi_paths[i, n]

            c_sim[i, n] = interp_c[idx](W, r, π_val)
            wN_sim[i, n] = interp_w[idx][1](W, r, π_val)
            wS_sim[i, n] = interp_w[idx][2](W, r, π_val)
        end
    end
    return c_sim, wN_sim, wS_sim
end

# ==============================================================================
# 1. DP Setup & Solve (Problem 3: Incomplete Market)
# ==============================================================================
println("Solving Problem 3 DP (Incomplete Market without Inflation-Linked Bonds)...")

M, dt, β, γ = 10, 1.0, 0.96, 5.0
u(x) = (x^(1 - γ)) / (1 - γ)

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

# Grids
G_w = 400
W_grid = generate_log_spaced_grid(0.5, 100.0, G_w)
Z_grids = [
    generate_linear_grid(0.0, 0.1, 10),  # r_grid (Center index 3 is 0.02)
    generate_linear_grid(0.0, 0.1, 10)   # π_grid (Center index 3 is 0.02)
]
c_grid = generate_linear_grid(0.01, 0.99, 10)

# ω = [ω_N, ω_S]
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

# Custom Budget Constraint with strict real income (O_t / Π_t = 1.0)
function problem3_budget_constraint(W, c, ω, R_e, R_base)
    income_real = 1.0 * dt
    return (1.0 - c) * W * (dot(ω, R_e) + R_base) + income_real
end

crra_ex = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

V, pol_c, pol_w = solve_dynamic_program(
    W_grid, Z_grids, c_grid, omega_space,
    ε_nodes, W_weights, transition_prob3,
    M, β, u, fractional_consumption,
    problem3_budget_constraint, crra_ex
)

interp_c, interp_w = create_policy_interpolators(pol_c, pol_w, W_grid, Z_grids)

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
    c   = interp_c[idx](W_safe, r_val, pi_val)

    # Real wealth drift with additive income
    return W_safe * (ω_N * (-λ_r * σ_r * B_r_val) + ω_S * (λ_S * σ_S) + r_val - pi_val - c) + 1.0
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
c_sim, wN_sim, wS_sim = extract_controls_3d(world.paths.W, world.paths.r, world.paths.pi, interp_c, interp_w, 1.0)

# ==============================================================================
# 3. Generating Plots
# ==============================================================================
println("Generating and saving plots...")

# Plot 1: Value Function vs Wealth
fig_v = Figure(size=(800, 400))
ax_v = Axis(fig_v[1,1], title="Problem 3: Value Function V(W) (at r=0.02, π=0.02)", xlabel="Real Wealth (W)", ylabel="Expected Utility")
lines!(ax_v, W_grid, V[:, 3, 3, 1], label="t = 1", linewidth=3, color=:dodgerblue)
lines!(ax_v, W_grid, V[:, 3, 3, 5], label="t = 5", linewidth=3, color=:darkorange)
lines!(ax_v, W_grid, V[:, 3, 3, 10], label="t = 10", linewidth=3, color=:firebrick)
axislegend(ax_v, position=:rb)
save("prob3_value_function.png", fig_v)

# Plot 2: Heatmap for Nominal Bond Weight (Fixing π = 0.02, t = 1)
fig_heat_N = Figure(size=(800, 600))
ax_N = Axis(fig_heat_N[1, 1], title="Nominal Bond Policy Heatmap (t=1, π=0.02)", xlabel="Real Wealth (W)", ylabel="Interest Rate (r)")
slice_N = [pol_w[w, r, 3, 1][1] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]
co_N = contourf!(ax_N, W_grid, Z_grids[1], slice_N, colormap=:viridis, levels=20)
Colorbar(fig_heat_N[1, 2], co_N, label="Nominal Bond Weight")
save("prob3_heatmap_nominal_bond.png", fig_heat_N)

# Plot 2: Heatmap for Nominal Bond Weight (Fixing π = 0.02, t = 1) #FIXXXXXX
fig_heat_N = Figure(size=(800, 600))
ax_N = Axis(fig_heat_N[1, 1], title="Nominal Bond Policy Heatmap (t=1, π=0.02)", xlabel="Inflation rate (π)", ylabel="Interest Rate (r)")
slice_N = [pol_w[w, r, 3, 1][1] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]
co_N = contourf!(ax_N, W_grid, Z_grids[1], slice_N, colormap=:viridis, levels=20)
Colorbar(fig_heat_N[1, 2], co_N, label="Nominal Bond Weight")
save("prob3_heatmap_nominal_bond.png", fig_heat_N)

# Plot 3: Heatmap for Stock Weight (Fixing π = 0.02, t = 1)
fig_heat_S = Figure(size=(800, 600))
ax_S = Axis(fig_heat_S[1, 1], title="Stock Policy Heatmap (t=1, π=0.02)", xlabel="Real Wealth (W)", ylabel="Interest Rate (r)")
slice_S = [pol_w[w, r, 3, 1][2] for w in 1:length(W_grid), r in 1:length(Z_grids[1])]
co_S = contourf!(ax_S, W_grid, Z_grids[1], slice_S, colormap=:plasma, levels=20)
Colorbar(fig_heat_S[1, 2], co_S, label="Stock Weight")
save("prob3_heatmap_stock.png", fig_heat_S)

# Plot 4: Simulated Paths Overlay (Real Wealth)
fig_paths_W = plot_paths_overlay(Matrix(world.paths.W); title="Problem 3: Simulated Real Wealth Paths", xlabel="Time (Years)", ylabel="Real Wealth", line_color=:dodgerblue)
save("prob3_sim_wealth_paths.png", fig_paths_W)

# Plot 5: Simulated Paths Overlay (Interest Rate & Inflation)
fig_paths_r = plot_paths_overlay(Matrix(world.paths.r); title="Problem 3: Simulated Interest Rate Paths", line_color=:firebrick)
save("prob3_sim_rate_paths.png", fig_paths_r)
fig_paths_pi = plot_paths_overlay(Matrix(world.paths.pi); title="Problem 3: Simulated Inflation Paths", line_color=:darkorange)
save("prob3_sim_inflation_paths.png", fig_paths_pi)

# Plot 6: Average Strategy Evolutions
fig_mean_c = plot_mean_with_bounds(c_sim; title="Mean Consumption Strategy", ylabel="Consumption Fraction", color=:purple)
save("prob3_mean_consumption.png", fig_mean_c)

fig_mean_wN = plot_mean_with_bounds(wN_sim; title="Mean Nominal Bond Allocation", ylabel="Portfolio Weight", color=:blue)
save("prob3_mean_nominal_bond.png", fig_mean_wN)

fig_mean_wS = plot_mean_with_bounds(wS_sim; title="Mean Stock Allocation", ylabel="Portfolio Weight", color=:green)
save("prob3_mean_stock.png", fig_mean_wS)


# Plot 7: Objective Curve (Ensures we aren't clipping the boundaries anymore!)
fig_obj = plot_objective_curve(wN_range, obj_curve_vals,
                               title="Bellman Objective vs Nominal Bond Weight (t=1, W=50)",
                               xlabel="Nominal Bond Weight (ω_N)")
save("prob3_objective_curve.png", fig_obj)

# Plot 8: Deterministic Target-Date Glidepath
times_seq = 1:M
glide_wN = [interp_w[t][1](50.0, 0.02, 0.02) for t in times_seq]
glide_wS = [interp_w[t][2](50.0, 0.02, 0.02) for t in times_seq]
fig_glide = plot_deterministic_glidepath(times_seq, glide_wN, glide_wS)
save("prob3_deterministic_glidepath.png", fig_glide)

# Plot 9: Wealth vs Human Capital Evolution
# Expected PV of future real income discounted at real rate (approx 0.0 since r_bar = pi_bar = 0.02)
H_mean = [(M - t + 1) * dt for t in times_seq]
W_mean = vec(mean(world.paths.W, dims=1))[1:M]
fig_wealth_comp = plot_wealth_composition(times_seq, W_mean, H_mean)
save("prob3_wealth_composition.png", fig_wealth_comp)

println("All Problem 3 DP solutions, simulations, and plots generated successfully!")