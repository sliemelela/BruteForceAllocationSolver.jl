using BruteForceAllocationSolver
using FinancialMarketSimulation
using FastGaussQuadrature
using CairoMakie
using LinearAlgebra
using Statistics
using Interpolations
using StaticArrays

println("==================================================")
println("Advanced Solvers: Problem 1 (Complete Market)")
println("==================================================")

# ==============================================================================
# 1. Global Parameters & Exact Analytical Baseline
# ==============================================================================
# M, dt, γ = 20, 0.5, 5.0
T = M * dt
γ_tilde = (γ - 1.0) / γ

# u(x) = (x^(1 - γ)) / (1 - γ)
# inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# Economic Parameters
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π, λ_π = 0.05, 0.02, 0.02, 0.05
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N, τ_I = 10.0, 10.0

ρ_rπ, ρ_rS, ρ_πS = 0.5, 0.5, 0.5
ρ_mat = [1.0 ρ_rπ ρ_rS; ρ_rπ 1.0 ρ_πS; ρ_rS ρ_πS 1.0]
lambda_vec = [λ_r, λ_π, λ_S]
phi_vec = -(ρ_mat \ lambda_vec)

# Helpers for Analytical CE & Durations
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

r_0, π_0, F_0 = 0.02, 0.02, 140.0
W_0 = 149.80 # Approximated Total Wealth initial state
exponent = B_r(T) * r_0 - B_π(T) * π_0 - E2(T) - γ_tilde * V2(T)
analytical_CE = W_0 * exp(exponent)

# ==============================================================================
# 2. Reduced Grids for Performance
# ==============================================================================
G_w = 50
W_grid = generate_log_spaced_grid(10.0, 300.0, G_w)

Z_grids = [
    generate_linear_grid(-0.02, 0.06, 10),
    generate_linear_grid(-0.02, 0.06, 10)
]

omega_space = SVector{3, Float64}[]
for w_N in range(0.0, 7.0, length=11)
    for w_I in range(0.0, 2.0, length=15)
        for w_S in range(0.0, 3.0, length=11)
            push!(omega_space, SVector(w_N, w_I, w_S))
        end
    end
end
ε_nodes, W_weights = generate_gaussian_shocks(3, 5, ρ_mat)

function make_problem1_transition(κ_r, θ_r, σ_r, λ_r, τ_N, κ_π, θ_π, σ_π, λ_π, τ_I, λ_S, σ_S, ρ_rπ, dt)
    B_r_N, B_r_I, B_π_I = B_r(τ_N), B_r(τ_I), B_π(τ_I)
    vol_N_r, vol_I_r, vol_I_π = -B_r_N * σ_r, -B_r_I * σ_r, B_π_I * σ_π
    var_N = vol_N_r^2
    var_I = vol_I_r^2 + vol_I_π^2 + 2 * ρ_rπ * vol_I_r * vol_I_π
    var_S = σ_S^2

    return function(Z, ε)
        r_n, π_n = Z[1], Z[2]
        ε_r, ε_π, ε_S = ε[1], ε[2], ε[3]

        r_next = clamp(r_n + κ_r * (overline_r - r_n) * dt + σ_r * sqrt(dt) * ε_r, -0.02, 0.06)
        π_next = clamp(π_n + κ_π * (overline_π - π_n) * dt + σ_π * sqrt(dt) * ε_π, -0.06, 0.10)
        Z_next = SVector(r_next, π_next)

        Rf_nom = exp(r_n * dt)
        R_N = exp((r_n - λ_r * σ_r * B_r_N - 0.5 * var_N) * dt + vol_N_r * sqrt(dt) * ε_r)
        R_I = exp((r_n - λ_r * σ_r * B_r_I + λ_π * σ_π * B_π_I - 0.5 * var_I) * dt + vol_I_r * sqrt(dt) * ε_r + vol_I_π * sqrt(dt) * ε_π)
        R_S = exp((r_n + λ_S * σ_S - 0.5 * var_S) * dt + σ_S * sqrt(dt) * ε_S)

        Re = SVector(R_N - Rf_nom, R_I - Rf_nom, R_S - Rf_nom)
        R_base_real = exp((r_n - π_n) * dt)
        return Z_next, Re, R_base_real
    end
end

transition_prob1 = make_problem1_transition(κ_r, overline_r, σ_r, λ_r, τ_N, κ_π, overline_π, σ_π, λ_π, τ_I, λ_S, σ_S, ρ_rπ, dt)

function problem1_budget_constraint(W, c, ω, R_e, R_base)
    return max(W * (dot(ω, R_e) + R_base), 1e-2)
end
crra_ex = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

# ==============================================================================
# 3. Solvers & Benchmarking
# ==============================================================================
solvers = [
    # Uncomment or comment whichever solvers you want to test!
    # ("BruteForce", BruteForceSolver()),
    # ("Zooming", ZoomingSolver(iterations=5, points_per_dim=4, zoom_range_factor=1.1)),
    ("Optim", OptimSolver(use_gradients=true, coarse_warm_start_n=5))
]

results = []
pol_w_best = nothing
V_best = nothing
best_CE = -Inf # Track the highest CE found to use for plotting

println("\nWarming up (pre-compiling) solvers...")
for (name, solver) in solvers
    solve_dynamic_program(solver, W_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob1, 1, u, identity, problem1_budget_constraint, crra_ex)
end

println("\nRunning Benchmarks (M = 10)...")
for (name, solver) in solvers
    println("  Running $name solver...")

    local V, pol_w # Declare locally so it survives the @elapsed block
    time_taken = @elapsed begin
        V, pol_w = solve_dynamic_program(solver, W_grid, Z_grids, omega_space, ε_nodes, W_weights, transition_prob1, M, u, identity, problem1_budget_constraint, crra_ex)
    end

    # Extract Certainty Equivalent
    V_interp = linear_interpolation((W_grid, Z_grids[1], Z_grids[2]), V[:, :, :, 1], extrapolation_bc=Line())
    CE_num = calculate_certainty_equivalent(V_interp(W_0, r_0, π_0), inv_u)

    # FIX: Dynamically save the best policy regardless of which solver it is
    global best_CE, pol_w_best, V_best
    if CE_num > best_CE
        best_CE = CE_num
        pol_w_best = pol_w
        V_best = V
    end

    push!(results, (name, time_taken, CE_num))
end

println("\n==================================================")
println("PERFORMANCE & ACCURACY REPORT")
println("==================================================")
for (name, time_taken, CE_num) in results
    println(rpad(name, 12), " | Time: ", rpad("$(round(time_taken, digits=2))s", 8),
            " | CE: ", rpad(round(CE_num, digits=4), 8))
end
println("==================================================")

# ==============================================================================
# 4. Forward Monte Carlo Simulation using Best Policy
# ==============================================================================
println("\nRunning Forward Monte Carlo Simulation with Optim Policy...")
dummy_pol_c = zeros(size(pol_w_best))
_, interp_w = create_policy_interpolators(dummy_pol_c, pol_w_best, W_grid, Z_grids)

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
B_r_N, B_r_I, B_pi_I = (1.0 - exp(-κ_r * τ_N)) / κ_r, (1.0 - exp(-κ_r * τ_I)) / κ_r, (1.0 - exp(-κ_π * τ_I)) / κ_π

drift_W1(t, W, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    W_safe = max(W, 1e-5)
    ω_N, ω_I, ω_S = interp_w[idx][1](W_safe, r_val, pi_val), interp_w[idx][2](W_safe, r_val, pi_val), interp_w[idx][3](W_safe, r_val, pi_val)
    RP_N, RP_I, RP_S = ω_N * (-λ_r * σ_r * B_r_N), ω_I * (-λ_r * σ_r * B_r_I + λ_π * σ_π * B_pi_I), ω_S * (λ_S * σ_S)
    return W_safe * (RP_N + RP_I + RP_S + r_val - pi_val)
end

diff_W1(t, W, r_val, pi_val) = begin
    idx = min(M, floor(Int, t/dt) + 1)
    W_safe = max(W, 1e-5)
    ω_N, ω_I, ω_S = interp_w[idx][1](W_safe, r_val, pi_val), interp_w[idx][2](W_safe, r_val, pi_val), interp_w[idx][3](W_safe, r_val, pi_val)
    return [W_safe * (-ω_N * B_r_N * σ_r - ω_I * B_r_I * σ_r), W_safe * (ω_I * B_pi_I * σ_π), W_safe * (ω_S * σ_S)]
end

w1_proc = GenericSDEProcess(:W, drift_W1, diff_W1, W_0, [1, 2, 3], [:r, :pi])
conf_prob1 = MarketConfig(sims=500, T=T, dt=dt, M=M, processes=[rate_proc, pi_proc, w1_proc], correlations=ρ_mat)
world_1 = build_world(conf_prob1)

wN_sim, wI_sim, wS_sim = extract_controls_prob1(world_1.paths.W, world_1.paths.r, world_1.paths.pi, interp_w, dt)

# ==============================================================================
# 5. Extract Transformed Weights (Equation 37)
# ==============================================================================
function get_HC_and_durations(t_step, r_val, pi_val)
    H_t, D_r_num, D_pi_num = 0.0, 0.0, 0.0
    t = (t_step - 1) * dt
    for step in t_step:M
        h = step * dt - t
        P_real = exp(A_I(h) - B_r(h)*r_val + B_π(h)*pi_val)
        H_t += P_real * dt
        D_r_num += P_real * B_r(h) * dt
        D_pi_num += P_real * B_π(h) * dt
    end
    if H_t < 1e-8 return 0.0, 0.0, 0.0 end
    return D_r_num / H_t, D_pi_num / H_t, H_t
end

function get_numerical_w_star(t_step, F_val, r_val, pi_val)
    t = (t_step - 1) * dt
    h = T - t
    if h < 1e-8 return 0.0, 0.0, 0.0 end
    D_r, D_pi, H_t = get_HC_and_durations(t_step, r_val, pi_val)
    W_t = F_val + H_t
    wN_tilde = interp_w[t_step][1](W_t, r_val, pi_val)
    wI_tilde = interp_w[t_step][2](W_t, r_val, pi_val)
    wS_tilde = interp_w[t_step][3](W_t, r_val, pi_val)
    wN_star = (W_t / F_val) * wN_tilde + (H_t / F_val) * (D_pi / B_π(h) - D_r / B_r(h))
    wI_star = (W_t / F_val) * wI_tilde - (H_t / F_val) * (D_pi / B_π(h))
    wS_star = (W_t / F_val) * wS_tilde
    return wN_star, wI_star, wS_star
end

# ==============================================================================
# 6. Full Plotting Suite
# ==============================================================================
println("Generating all plots...")
fixed_w_idx = argmin(abs.(W_grid .- 150.0))
fixed_W_val = round(W_grid[fixed_w_idx], digits=2)
fixed_r_idx = 3; fixed_pi_idx = 3

# Plot 1 & 2: Value & CE
save("prob1_adv_value_function.png", plot_curves(W_grid, [V_best[:, fixed_r_idx, fixed_pi_idx, 1], V_best[:, fixed_r_idx, fixed_pi_idx, 5], V_best[:, fixed_r_idx, fixed_pi_idx, 10]], ["t = 1", "t = 5", "t = 10"]; title="Expected Utility V(W) (r=0.02, π=0.02)", xlabel="Total Real Wealth", ylabel="Utility", legend_pos=:rb))
ce_over_time = [calculate_certainty_equivalent(V_best[fixed_w_idx, fixed_r_idx, fixed_pi_idx, t], inv_u) for t in 1:M]
save("prob1_adv_ce_progression.png", plot_curves(1:M, [ce_over_time], ["CE Total Wealth"]; title="CE Progression (W=$fixed_W_val, r=0.02, π=0.02)", xlabel="Time Step (t)", ylabel="Guaranteed Terminal Wealth", legend_pos=:rt))

# Plot 3-5: MC Mean Strategy
save("prob1_adv_mean_nominal_bond.png", plot_mean_with_bounds(wN_sim; title="Mean Nominal Bond Allocation", ylabel="Weight", color=:blue))
save("prob1_adv_mean_ilb.png", plot_mean_with_bounds(wI_sim; title="Mean ILB Allocation", ylabel="Weight", color=:purple))
save("prob1_adv_mean_stock.png", plot_mean_with_bounds(wS_sim; title="Mean Stock Allocation", ylabel="Weight", color=:green))

# Plot 6-8: Transformed Heatmaps (Financial Wealth w-star) at t=0
slice_N_star = [get_numerical_w_star(1, F_0, r, pi)[1] for r in Z_grids[1], pi in Z_grids[2]]
slice_I_star = [get_numerical_w_star(1, F_0, r, pi)[2] for r in Z_grids[1], pi in Z_grids[2]]
slice_S_star = [get_numerical_w_star(1, F_0, r, pi)[3] for r in Z_grids[1], pi in Z_grids[2]]

save("prob1_adv_heatmap_N_star.png", plot_heatmap(Z_grids[1], Z_grids[2], slice_N_star; title="Nominal Bond w* (F=$F_0, t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("prob1_adv_heatmap_I_star.png", plot_heatmap(Z_grids[1], Z_grids[2], slice_I_star; title="ILB w* (F=$F_0, t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("prob1_adv_heatmap_S_star.png", plot_heatmap(Z_grids[1], Z_grids[2], slice_S_star; title="Stock w* (F=$F_0, t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

println("All Problem 1 plots and evaluations complete!")