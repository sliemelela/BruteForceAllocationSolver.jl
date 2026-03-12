using BruteForceAllocationSolver
using FinancialMarketSimulation
using FastGaussQuadrature
using CairoMakie
using LinearAlgebra

# ==============================================================================
# HELPER: Extract Controls from Simulated Paths
# ==============================================================================
function extract_controls_1d(W_paths, interp_c, interp_w, dt)
    sims, steps = size(W_paths)
    c_sim = zeros(sims, steps)
    w_sim = zeros(sims, steps)

    for n in 1:steps
        idx = min(length(interp_c), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            W = max(W_paths[i, n], 1e-5) # Prevent bounds errors
            c_sim[i, n] = interp_c[idx](W)
            w_sim[i, n] = interp_w[idx][1](W)
        end
    end
    return c_sim, w_sim
end

function extract_controls_2d(W_paths, r_paths, interp_c, interp_w, dt)
    sims, steps = size(W_paths)
    c_sim = zeros(sims, steps)
    w_sim = zeros(sims, steps)

    for n in 1:steps
        idx = min(length(interp_c), floor(Int, (n-1)*dt/dt) + 1)
        for i in 1:sims
            W = max(W_paths[i, n], 1e-5)
            r = r_paths[i, n]
            c_sim[i, n] = interp_c[idx](W, r)
            w_sim[i, n] = interp_w[idx][1](W, r)
        end
    end
    return c_sim, w_sim
end

# ==============================================================================
# TEST CASE 1: Standard Merton Model
# ==============================================================================
println("Solving & Simulating Test Case 1 (Merton)...")

# 1. DP Setup & Solve
M, dt, β, γ = 10, 1.0, 0.96, 5.0
u(x) = (x^(1 - γ))/(1 - γ)
W_grid = generate_log_spaced_grid(1.0, 100.0, 200)
c_grid = generate_linear_grid(0.01, 0.99, 50)
omega_space = [[ω] for ω in generate_linear_grid(0.0, 1.0, 101)]

r, μ, σ = 0.02, 0.07, 0.20
trans_merton = make_merton_transition(r, μ, σ, dt)
crra_ex = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

ρ_mat1 = fill(1.0, 1, 1)
ε_nodes1, W_weights1 = generate_gaussian_shocks(1, 10, ρ_mat1)

V, pol_c, pol_w = solve_dynamic_program(
    W_grid, Vector{Float64}[], c_grid, omega_space,
    ε_nodes1, W_weights1,
    trans_merton, M, β, u, fractional_consumption,
    standard_budget_constraint, crra_ex
)

interp_c, interp_w = create_policy_interpolators(
    pol_c, pol_w, W_grid, Vector{Float64}[]
)

# 2. Forward Simulation via SDE (Baseline)
drift_W(t, W) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    return W * (interp_w[idx][1](W)*(μ - r) + r - interp_c[idx](W))
end
diff_W(t, W) = W * interp_w[min(M, floor(Int, t/dt)+1)][1](W) * σ

wealth_base = GenericSDEProcess(:W, drift_W, diff_W, 50.0, [1])
conf_base = MarketConfig(sims=500, T=10.0, dt=1.0, M=10,
                         processes=[wealth_base])
world_base = build_world(conf_base)

# 3. Forward Simulation via SDE (Shocked Scenario)
diff_W_shock(t, W) = begin
    actual_σ = t >= 5.0 ? 0.40 : 0.20
    return W * interp_w[min(M, floor(Int, t/dt)+1)][1](W) * actual_σ
end

wealth_shock = GenericSDEProcess(:W, drift_W, diff_W_shock, 50.0, [1])
conf_shock = MarketConfig(sims=500, T=10.0, dt=1.0, M=10,
                          processes=[wealth_shock])
world_shock = build_world(conf_shock)

# 4. Extract Controls & Plot Everything for TC1
c_base, w_base = extract_controls_1d(
    world_base.paths.W, interp_c, interp_w, 1.0
)
c_shck, w_shck = extract_controls_1d(
    world_shock.paths.W, interp_c, interp_w, 1.0
)

println("  > Generating TC1 Plots...")
fig_v1 = plot_value_function(V, W_grid, [1, 5, 10];
                             title="Merton: Expected Utility V(W)")
save("tc1_value_function.png", fig_v1)

pol_w_scalar = map(x -> x[1], pol_w)
fig_pol_w1 = plot_policy_vs_state(pol_w_scalar, W_grid, Vector{Float64}[], 1;
                                  plot_against_W=true, ylabel="Risky Weight")
save("tc1_policy_vs_state.png", fig_pol_w1)

fig_paths1 = plot_paths_overlay(Matrix(world_base.paths.W);
                                title="Merton: Simulated Wealth Paths")
save("tc1_wealth_paths.png", fig_paths1)

fig_mean1 = plot_mean_with_bounds(c_base;
                                  title="Merton: Mean Consumption",
                                  ylabel="Fraction", color=:purple)
save("tc1_consumption_mean.png", fig_mean1)

fig_c1 = plot_shock_comparison(c_base, c_shck; shock_time=5,
                               title="Merton: Consumption (Vol Shock)",
                               ylabel="Consumption Fraction")
fig_w1 = plot_shock_comparison(w_base, w_shck; shock_time=5,
                               title="Merton: Portfolio Weight (Vol Shock)",
                               ylabel="Risky Weight")
save("tc1_consumption_shock.png", fig_c1)
save("tc1_investment_shock.png", fig_w1)


# ==============================================================================
# TEST CASE 2: Stochastic Interest Rate Model (Constant Mu)
# ==============================================================================
println("Solving & Simulating Test Case 2 (Stochastic Rates - Constant Mu)...")

# 1. DP Setup & Solve
Z_grids = [generate_linear_grid(0.0, 0.10, 11)]
κ, θ, σ_r = 0.1, 0.03, 0.01

trans_stoch = make_stochastic_r_constant_mu_transition(
    κ, θ, σ_r, μ, σ, 0.0, dt
)

ρ_mat2 = [1.0 0.0; 0.0 1.0]
ε_nodes2, W_weights2 = generate_gaussian_shocks(2, 8, ρ_mat2)

V2, pol_c2, pol_w2 = solve_dynamic_program(
    W_grid, Z_grids, c_grid, omega_space,
    ε_nodes2, W_weights2,
    trans_stoch, M, β, u, fractional_consumption,
    standard_budget_constraint, crra_ex
)

interp_c2, interp_w2 = create_policy_interpolators(
    pol_c2, pol_w2, W_grid, Z_grids
)

# 2. Forward Simulation via SDEs
rate_proc = VasicekProcess(:r, κ, θ, σ_r, 0.05, 1)

drift_W2(t, W, r_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    return W * (interp_w2[idx][1](W, r_val)*(μ - r_val) +
                r_val - interp_c2[idx](W, r_val))
end
diff_W2(t, W, r_val) = W * interp_w2[min(M, floor(Int, t/dt)+1)][1](W,r_val) * σ
w2_base = GenericSDEProcess(:W, drift_W2, diff_W2, 50.0, [2], [:r])

conf_base2 = MarketConfig(sims=500, T=10.0, dt=1.0, M=10,
                          processes=[rate_proc, w2_base],
                          correlations=ρ_mat2)
world_base2 = build_world(conf_base2)

conf_shock2 = MarketConfig(sims=500, T=10.0, dt=1.0, M=10,
                           processes=[rate_proc, w2_base],
                           correlations=ρ_mat2)
world_shock2 = build_world(conf_shock2)

# ---> INSTANT POST-SIMULATION SHOCK <---
world_shock2.paths.r[:, 6:end] .+= 0.05

# 3. Extract & Plot Everything for TC2
c_base2, w_base2 = extract_controls_2d(
    world_base2.paths.W, world_base2.paths.r, interp_c2, interp_w2, 1.0
)
c_shck2, w_shck2 = extract_controls_2d(
    world_shock2.paths.W, world_shock2.paths.r, interp_c2, interp_w2, 1.0
)

println("  > Generating TC2 Plots...")
pol_w2_scalar = map(x -> x[1], pol_w2)
fig_heat2 = plot_policy_heatmap(pol_w2_scalar, W_grid, Z_grids, 1;
                                title="Constant Mu: Portfolio Weight Heatmap",
                                xlabel="Wealth (W)", ylabel="Interest Rate (r)")
save("tc2_policy_heatmap.png", fig_heat2)

fig_r_paths2 = plot_paths_overlay(Matrix(world_base2.paths.r);
                                  title="Stoch Rates: Simulated Interest Rate Paths",
                                  ylabel="Rate (r)", line_color=:firebrick)
save("tc2_rate_paths.png", fig_r_paths2)

fig_c2 = plot_shock_comparison(c_base2, c_shck2; shock_time=5,
                               title="Constant Mu: Consumption (Post-Sim Rate Spike)",
                               ylabel="Consumption Fraction")
fig_w2 = plot_shock_comparison(w_base2, w_shck2; shock_time=5,
                               title="Constant Mu: Portfolio (Post-Sim Rate Spike)",
                               ylabel="Risky Weight")
save("tc2_consumption_shock.png", fig_c2)
save("tc2_investment_shock.png", fig_w2)


# ==============================================================================
# TEST CASE 3: Stochastic Interest Rate Model (Constant Risk Premium)
# ==============================================================================
println("Solving & Simulating Test Case 3 (Constant Risk Premium)...")

# 1. DP Setup & Solve
# To keep initial conditions comparable to TC2:
# In TC2, μ = 0.07, initial r ≈ 0.02 -> premium ≈ 0.05.
# With σ = 0.20, we need λ_S = 0.25 so that λ_S * σ = 0.05.
λ_S = 0.25

trans_premium = make_stochastic_r_constant_premium_transition(
    κ, θ, σ_r, λ_S, σ, 0.0, dt
)

V3, pol_c3, pol_w3 = solve_dynamic_program(
    W_grid, Z_grids, c_grid, omega_space,
    ε_nodes2, W_weights2, # Reuse identical 2D nodes from TC2
    trans_premium, M, β, u, fractional_consumption,
    standard_budget_constraint, crra_ex
)

interp_c3, interp_w3 = create_policy_interpolators(
    pol_c3, pol_w3, W_grid, Z_grids
)

# 2. Forward Simulation via SDEs
# Baseline Wealth Drift
# Expected asset return is strictly r_val + λ_S * σ.
# Thus, excess return applied to portfolio weight is exactly λ_S * σ.
drift_W3(t, W, r_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    return W * (interp_w3[idx][1](W, r_val)*(λ_S * σ) +
                r_val - interp_c3[idx](W, r_val))
end
diff_W3(t, W, r_val) = W * interp_w3[min(M, floor(Int, t/dt)+1)][1](W,r_val) * σ

w3_base = GenericSDEProcess(:W, drift_W3, diff_W3, 50.0, [2], [:r])

conf_base3 = MarketConfig(sims=500, T=10.0, dt=1.0, M=10,
                          processes=[rate_proc, w3_base], # Reuse rate_proc from TC2
                          correlations=ρ_mat2)
world_base3 = build_world(conf_base3)

conf_shock3 = MarketConfig(sims=500, T=10.0, dt=1.0, M=10,
                           processes=[rate_proc, w3_base],
                           correlations=ρ_mat2)
world_shock3 = build_world(conf_shock3)

# ---> INSTANT POST-SIMULATION SHOCK <---
world_shock3.paths.r[:, 6:end] .+= 0.05

# 3. Extract & Plot Everything for TC3
c_base3, w_base3 = extract_controls_2d(
    world_base3.paths.W, world_base3.paths.r, interp_c3, interp_w3, 1.0
)
c_shck3, w_shck3 = extract_controls_2d(
    world_shock3.paths.W, world_shock3.paths.r, interp_c3, interp_w3, 1.0
)

println("  > Generating TC3 Plots...")
pol_w3_scalar = map(x -> x[1], pol_w3)
fig_heat3 = plot_policy_heatmap(pol_w3_scalar, W_grid, Z_grids, 1;
                                title="Constant Premium: Portfolio Weight Heatmap",
                                xlabel="Wealth (W)", ylabel="Interest Rate (r)")
save("tc3_policy_heatmap.png", fig_heat3)

fig_c3 = plot_shock_comparison(c_base3, c_shck3; shock_time=5,
                               title="Constant Premium: Consumption (Post-Sim Rate Spike)",
                               ylabel="Consumption Fraction")
fig_w3 = plot_shock_comparison(w_base3, w_shck3; shock_time=5,
                               title="Constant Premium: Portfolio (Post-Sim Rate Spike)",
                               ylabel="Risky Weight")
save("tc3_consumption_shock.png", fig_c3)
save("tc3_investment_shock.png", fig_w3)

println("All simulations complete and plots saved!")