using BruteForceAllocationSolver
using FinancialMarketSimulation
using FastGaussQuadrature
using CairoMakie
using LinearAlgebra
using Statistics
using Interpolations
using StaticArrays

println("==================================================")
println("Setting up Problem 1 (Complete Market) Master Script")
println("==================================================")

# ==============================================================================
# 1. Global Parameters & Market Prices of Risk
# ==============================================================================
# M, dt, γ = 10, 1.0, 5.0
T = M * dt
γ_tilde = (γ - 1.0) / γ

# u(x) = (x^(1 - γ)) / (1 - γ)
# inv_u(v) = ((1.0 - γ) * v)^(1.0 / (1.0 - γ))

# Economic Parameters
κ_r, overline_r, σ_r, λ_r = 0.1, 0.02, 0.01, -0.1
κ_π, overline_π, σ_π, λ_π = 0.05, 0.02, 0.02, 0.05
a, b, σ_S, λ_S = 1.0, 0.0, 0.20, 0.25
τ_N = 10.0 # Nominal bond maturity
τ_I = 10.0 # Inflation-linked bond maturity

# Correlation Matrix
ρ_rπ, ρ_rS, ρ_πS = 0.5, -0.2, -0.2
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
# 2. Exact Analytical Baseline & Helper Functions
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
F_0 = 5.0
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
# 3. Generating Analytical Closed-Form Plots
# ==============================================================================
println("\nGenerating Analytical Closed-Form Plots...")

# Equation (53): Analytical Value Function
function eq53_value_function(t, W, r, pi_val)
    h = τ_N
    exponent = (γ - 1.0) * (E2(h) + γ_tilde * V2(h) - B_r(h)*r + B_π(h)*pi_val)
    return (W^(1 - γ)) / (1 - γ) * exp(exponent)
end

# Equation (55): Analytical Certainty Equivalent
function eq55_certainty_equivalent(t, W, r, pi_val)
    h = τ_N
    exponent = B_r(h)*r - B_π(h)*pi_val - E2(h) - γ_tilde * V2(h)
    return W * exp(exponent)
end

# Equation (143) helper: Real Zero-Coupon Bond Price (DI_t(s))
function DI_price(t, s, r, pi_val)
    h = s - t
    return exp(A_I(h) - B_r(h)*r + B_π(h)*pi_val)
end

# Equation (143): Duration of Human Capital
function eq143_durations(t, r, pi_val)
    H_t = 0.0
    D_r_num = 0.0
    D_pi_num = 0.0

    for step in Int(t+1):M
        s = step * dt
        h = s - t
        P_real = DI_price(t, s, r, pi_val)
        income = 1.0 * dt

        H_t += income * P_real
        D_r_num += income * P_real * B_r(h)
        D_pi_num += income * P_real * B_π(h)
    end

    if H_t < 1e-8 return 0.0, 0.0, 0.0 end
    return D_r_num / H_t, D_pi_num / H_t, H_t
end

# Equation (36): Optimal Portfolio WITHOUT Human Capital
function eq36_optimal_weights_no_hc(t)
    h = τ_N
    if h < 1e-8 return 0.0, 0.0, 0.0 end

    wN = (B_r(h) * σ_r * phi_vec[2] + B_π(h) * σ_π * phi_vec[1]) / (B_r(h) * B_π(h) * σ_r * σ_π)
    wI = (1.0 - 1.0/γ) - (phi_vec[2] / (B_π(h) * σ_π))
    wS = -phi_vec[3] / (γ * σ_S)
    return wN, wI, wS
end

# Equation (37): Optimal Portfolio WITH Human Capital
function eq37_optimal_weights_with_hc(t, F, r, pi_val)
    h = τ_N
    if h < 1e-8 return 0.0, 0.0, 0.0 end

    wN_tilde, wI_tilde, wS_tilde = eq36_optimal_weights_no_hc(t)
    D_r, D_pi, H_t = eq143_durations(t, r, pi_val)
    W_t = F + H_t

    wN_star = (W_t / F) * wN_tilde + (H_t / F) * (D_pi / B_π(h) - D_r / B_r(h))
    wI_star = (W_t / F) * wI_tilde - (H_t / F) * (D_pi / B_π(h))
    wS_star = (W_t / F) * wS_tilde

    return wN_star, wI_star, wS_star
end

# Generate Plot Data Arrays
t_seq = 0:1:9
ana_r_grid = range(-0.02, 0.06, length=50)
ana_pi_grid = range(-0.06, 0.10, length=50)
fixed_W_ana = 150.0

val_time = [eq53_value_function(t, fixed_W_ana, r_0, π_0) for t in t_seq]
ce_time = [eq55_certainty_equivalent(t, fixed_W_ana, r_0, π_0) for t in t_seq]

durations = [eq143_durations(t, r_0, π_0) for t in t_seq]
Dr_time, Dpi_time = [d[1] for d in durations], [d[2] for d in durations]

w_no_hc = [eq36_optimal_weights_no_hc(t) for t in t_seq]
wN_no_hc_time, wI_no_hc_time, wS_no_hc_time = [w[1] for w in w_no_hc], [w[2] for w in w_no_hc], [w[3] for w in w_no_hc]

w_hc = [eq37_optimal_weights_with_hc(t, F_0, r_0, π_0) for t in t_seq]
wN_hc_time, wI_hc_time, wS_hc_time = [w[1] for w in w_hc], [w[2] for w in w_hc], [w[3] for w in w_hc]

t_fix = 0.0
val_heat = [eq53_value_function(t_fix, fixed_W_ana, r, pi) for r in ana_r_grid, pi in ana_pi_grid]
ce_heat = [eq55_certainty_equivalent(t_fix, fixed_W_ana, r, pi) for r in ana_r_grid, pi in ana_pi_grid]
Dr_heat = [eq143_durations(t_fix, r, pi)[1] for r in ana_r_grid, pi in ana_pi_grid]
Dpi_heat = [eq143_durations(t_fix, r, pi)[2] for r in ana_r_grid, pi in ana_pi_grid]
wN_no_hc_heat = [eq36_optimal_weights_no_hc(t_fix)[1] for r in ana_r_grid, pi in ana_pi_grid]
wI_no_hc_heat = [eq36_optimal_weights_no_hc(t_fix)[2] for r in ana_r_grid, pi in ana_pi_grid]
wS_no_hc_heat = [eq36_optimal_weights_no_hc(t_fix)[3] for r in ana_r_grid, pi in ana_pi_grid]
wN_hc_heat = [eq37_optimal_weights_with_hc(t_fix, F_0, r, pi)[1] for r in ana_r_grid, pi in ana_pi_grid]
wI_hc_heat = [eq37_optimal_weights_with_hc(t_fix, F_0, r, pi)[2] for r in ana_r_grid, pi in ana_pi_grid]
wS_hc_heat = [eq37_optimal_weights_with_hc(t_fix, F_0, r, pi)[3] for r in ana_r_grid, pi in ana_pi_grid]

# Save Analytical Plots
save("analytical_eq53_value_over_time.png", plot_curves(t_seq, [val_time], ["V(W)"]; title="Eq 53: Value Function (W=150, r=0.02, π=0.02)", xlabel="Time", ylabel="Utility", legend_pos=:rt))
save("analytical_eq55_ce_over_time.png", plot_curves(t_seq, [ce_time], ["CE"]; title="Eq 55: Certainty Equivalent (W=150, r=0.02, π=0.02)", xlabel="Time", ylabel="CE", legend_pos=:rt))
save("analytical_eq143_durations_over_time.png", plot_curves(t_seq, [Dr_time, Dpi_time], ["D^r", "D^π"]; title="Eq 143: HC Sensitivities (r=0.02, π=0.02)", xlabel="Time", ylabel="Duration", legend_pos=:rt))
save("analytical_eq36_weights_over_time.png", plot_curves(t_seq, [wN_no_hc_time, wI_no_hc_time, wS_no_hc_time], ["w_N", "w_I", "w_S"]; title="Eq 36: Weights without HC", xlabel="Time", ylabel="Weight", legend_pos=:lt))
save("analytical_eq37_weights_over_time.png", plot_curves(t_seq, [wN_hc_time, wI_hc_time, wS_hc_time], ["w_N^*", "w_I^*", "w_S^*"]; title="Eq 37: Weights with HC (F=$F_0, r=0.02, π=0.02)", xlabel="Time", ylabel="Weight", legend_pos=:lt))

save("analytical_eq53_value_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, val_heat; title="Eq 53: Value Function (t=0, W=150)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:viridis, label="Utility"))
save("analytical_eq55_ce_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, ce_heat; title="Eq 55: Certainty Equivalent (t=0, W=150)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:viridis, label="CE"))
save("analytical_eq143_Dr_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, Dr_heat; title="Eq 143: D^r Sensitivity (t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="D^r"))
save("analytical_eq143_Dpi_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, Dpi_heat; title="Eq 143: D^π Sensitivity (t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="D^π"))
save("analytical_eq36_wN_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, wN_no_hc_heat; title="Eq 36: Nominal Bond WITHOUT HC (t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq36_wN_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, wN_no_hc_heat;
     title="Eq 36: Nominal Bond WITHOUT HC (t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

save("analytical_eq36_wI_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, wI_no_hc_heat;
     title="Eq 36: ILB WITHOUT HC (t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

save("analytical_eq36_wS_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, wS_no_hc_heat;
     title="Eq 36: Stock WITHOUT HC (t=0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq37_wN_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, wN_hc_heat; title="Eq 37: Nominal Bond WITH HC (t=0, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq37_wI_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, wI_hc_heat; title="Eq 37: ILB WITH HC (t=0, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq37_wS_heatmap.png", plot_heatmap(ana_r_grid, ana_pi_grid, wS_hc_heat; title="Eq 37: Stock WITH HC (t=0, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

# --- Heatmap Data (at t=5) ---
t_fix_5 = 5.0
wN_hc_heat_5 = [eq37_optimal_weights_with_hc(t_fix_5, F_0, r, pi)[1] for r in ana_r_grid, pi in ana_pi_grid]
wI_hc_heat_5 = [eq37_optimal_weights_with_hc(t_fix_5, F_0, r, pi)[2] for r in ana_r_grid, pi in ana_pi_grid]
wS_hc_heat_5 = [eq37_optimal_weights_with_hc(t_fix_5, F_0, r, pi)[3] for r in ana_r_grid, pi in ana_pi_grid]

# --- Heatmap Data (at t=9) ---
t_fix_9 = 9.0
wN_hc_heat_9 = [eq37_optimal_weights_with_hc(t_fix_9, F_0, r, pi)[1] for r in ana_r_grid, pi in ana_pi_grid]
wI_hc_heat_9 = [eq37_optimal_weights_with_hc(t_fix_9, F_0, r, pi)[2] for r in ana_r_grid, pi in ana_pi_grid]
wS_hc_heat_9 = [eq37_optimal_weights_with_hc(t_fix_9, F_0, r, pi)[3] for r in ana_r_grid, pi in ana_pi_grid]

# Save Analytical Heatmaps for t=5
save("analytical_eq37_wN_heatmap_t5.png", plot_heatmap(ana_r_grid, ana_pi_grid, wN_hc_heat_5; title="Eq 37: Nominal Bond WITH HC (t=5, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq37_wI_heatmap_t5.png", plot_heatmap(ana_r_grid, ana_pi_grid, wI_hc_heat_5; title="Eq 37: ILB WITH HC (t=5, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq37_wS_heatmap_t5.png", plot_heatmap(ana_r_grid, ana_pi_grid, wS_hc_heat_5; title="Eq 37: Stock WITH HC (t=5, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

# Save Analytical Heatmaps for t=9
save("analytical_eq37_wN_heatmap_t9.png", plot_heatmap(ana_r_grid, ana_pi_grid, wN_hc_heat_9; title="Eq 37: Nominal Bond WITH HC (t=9, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq37_wI_heatmap_t9.png", plot_heatmap(ana_r_grid, ana_pi_grid, wI_hc_heat_9; title="Eq 37: ILB WITH HC (t=9, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))
save("analytical_eq37_wS_heatmap_t9.png", plot_heatmap(ana_r_grid, ana_pi_grid, wS_hc_heat_9; title="Eq 37: Stock WITH HC (t=9, F=$F_0)", xlabel="Interest Rate (r)", ylabel="Inflation (π)", colormap=:plasma, label="Weight"))

# ==============================================================================
# 6. Additional Lifecycle & Shock Analysis Plots (ANALYTICAL)
# ==============================================================================
println("\nGenerating Lifecycle Strategy and Economic Shock Plots (Analytical)...")

# --- A. Investment Strategy Over Time (Combined Lifecycle Plot) ---
# We already calculated the baseline weights in Section 3: wN_hc_time, wI_hc_time, wS_hc_time
# Let's calculate the cash allocation: 1 - w_N - w_I - w_S
wC_hc_time = 1.0 .- wN_hc_time .- wI_hc_time .- wS_hc_time

save("prob1_analytic_adv_lifecycle_strategy.png",
    plot_curves(t_seq,
                [wN_hc_time, wI_hc_time, wS_hc_time, wC_hc_time],
                ["Nominal Bond (w_N)", "ILB (w_I)", "Stock (w_S)", "Cash (w_C)"];
                title="Analytical Investment Strategy Over Time",
                xlabel="Time Step (t)",
                ylabel="Portfolio Weight",
                legend_pos=:lt)
)

# --- B. Interest Rate Shock at t = 5 ---
# Create a deterministic shocked path for r: base rate before t=5, shocked by +5% after
r_shocked_path = [t < 5 ? r_0 : r_0 + 0.05 for t in t_seq]

# Calculate the new exact optimal weights given the shocked interest rate
w_hc_r_shock = [eq37_optimal_weights_with_hc(t_seq[i], F_0, r_shocked_path[i], π_0) for i in 1:length(t_seq)]
wN_hc_r_shock = [w[1] for w in w_hc_r_shock]
wI_hc_r_shock = [w[2] for w in w_hc_r_shock]

save("prob1_analytic_adv_shock_interest_rate_N.png",
    plot_curves(t_seq,
                [wN_hc_time, wN_hc_r_shock],
                ["Baseline w_N", "Shocked w_N (+5% r at t=5)"];
                title="Impact of Interest Rate Shock on Nominal Bond Allocation",
                xlabel="Time Step (t)",
                ylabel="Nominal Bond Weight",
                legend_pos=:lt)
)

save("prob1_analytic_adv_shock_interest_rate_I.png",
    plot_curves(t_seq,
                [wI_hc_time, wI_hc_r_shock],
                ["Baseline w_I", "Shocked w_I (+5% r at t=5)"];
                title="Impact of Interest Rate Shock on ILB Allocation",
                xlabel="Time Step (t)",
                ylabel="ILB Weight",
                legend_pos=:lt)
)

# --- C. Inflation Rate Shock at t = 5 ---
# Create a deterministic shocked path for pi: base rate before t=5, shocked by +5% after
pi_shocked_path = [t < 5 ? π_0 : π_0 + 0.05 for t in t_seq]

# Calculate the new exact optimal weights given the shocked inflation rate
w_hc_pi_shock = [eq37_optimal_weights_with_hc(t_seq[i], F_0, r_0, pi_shocked_path[i]) for i in 1:length(t_seq)]
wN_hc_pi_shock = [w[1] for w in w_hc_pi_shock]
wI_hc_pi_shock = [w[2] for w in w_hc_pi_shock]

save("prob1_analytic_adv_shock_inflation_N.png",
    plot_curves(t_seq,
                [wN_hc_time, wN_hc_pi_shock],
                ["Baseline w_N", "Shocked w_N (+5% π at t=5)"];
                title="Impact of Inflation Shock on Nominal Bond Allocation",
                xlabel="Time Step (t)",
                ylabel="Nominal Bond Weight",
                legend_pos=:lt)
)

save("prob1_analytic_adv_shock_inflation_I.png",
    plot_curves(t_seq,
                [wI_hc_time, wI_hc_pi_shock],
                ["Baseline w_I", "Shocked w_I (+5% π at t=5)"];
                title="Impact of Inflation Shock on ILB Allocation",
                xlabel="Time Step (t)",
                ylabel="ILB Weight",
                legend_pos=:lt)
)


# ==============================================================================
# 7. Financial Wealth Portfolio Strategy (ω^F) Plots
# ==============================================================================
println("\nGenerating Financial Wealth Portfolio Strategy Plots...")

# Calculate ω^F for the time sequence
wN_F_time = zeros(length(t_seq))
wI_F_time = zeros(length(t_seq))
wS_F_time = zeros(length(t_seq))

for i in 1:length(t_seq)
    t = t_seq[i]
    D_r, D_pi, H_t = eq143_durations(t, r_0, π_0)
    W_ratio = (F_0 + H_t) / F_0
    H_ratio = H_t / F_0

    # Merton Baseline at this time (could be constant or horizon-dependent)
    wN_m, wI_m, wS_m = eq36_optimal_weights_no_hc(t)

    # ω^F Calculation
    wN_F_time[i] = W_ratio * wN_m + H_ratio * (D_pi / B_π(τ_N) - D_r / B_r(τ_N))
    wI_F_time[i] = W_ratio * wI_m - H_ratio * (D_pi / B_π(τ_I))
    wS_F_time[i] = W_ratio * wS_m
end

# A. Lifecycle Plot: Financial Wealth Weights (ω^F)
save("analytical_prob1_FW_weights_over_time.png",
    plot_curves(t_seq,
                [wN_F_time, wI_F_time, wS_F_time],
                ["w_N (Nominal)", "w_I (ILB)", "w_S (Stock)"];
                title="Analytical Strategy in Financial Wealth Terms (F=$F_0)",
                xlabel="Time", ylabel="Financial Weight (ω^F)", legend_pos=:lt)
)

# B. State Heatmaps: Financial Wealth Weights (ω^F) at t=0
# We iterate over r and pi, calculating the H-adjusted weight for each state
wN_F_heat = [((F_0 + eq143_durations(0.0, r, pi)[3])/F_0) * eq36_optimal_weights_no_hc(0.0)[1] +
             (eq143_durations(0.0, r, pi)[3]/F_0) * (eq143_durations(0.0, r, pi)[2]/B_π(τ_N) - eq143_durations(0.0, r, pi)[1]/B_r(τ_N))
             for r in ana_r_grid, pi in ana_pi_grid]

wI_F_heat = [((F_0 + eq143_durations(0.0, r, pi)[3])/F_0) * eq36_optimal_weights_no_hc(0.0)[2] -
             (eq143_durations(0.0, r, pi)[3]/F_0) * (eq143_durations(0.0, r, pi)[2]/B_π(τ_I))
             for r in ana_r_grid, pi in ana_pi_grid]

wS_F_heat = [((F_0 + eq143_durations(0.0, r, pi)[3])/F_0) * eq36_optimal_weights_no_hc(0.0)[3]
             for r in ana_r_grid, pi in ana_pi_grid]

save("analytical_prob1_FW_heatmap_wN.png", plot_heatmap(ana_r_grid, ana_pi_grid, wN_F_heat; title="Analytical w_N^F (F=$F_0, t=0)", xlabel="r", ylabel="π", colormap=:plasma))
save("analytical_prob1_FW_heatmap_wI.png", plot_heatmap(ana_r_grid, ana_pi_grid, wI_F_heat; title="Analytical w_I^F (F=$F_0, t=0)", xlabel="r", ylabel="π", colormap=:plasma))
save("analytical_prob1_FW_heatmap_wS.png", plot_heatmap(ana_r_grid, ana_pi_grid, wS_F_heat; title="Analytical w_S^F (F=$F_0, t=0)", xlabel="r", ylabel="π", colormap=:plasma))

println("Analytical Financial Wealth strategy plots saved successfully!")


# ==============================================================================
# 2b. Evaluate Total Wealth & Certainty Equivalent Ratios
# ==============================================================================
# Calculate baseline exponent for CE (independent of wealth)
exponent = B_r(T) * r_0 - B_π(T) * π_0 - E2(T) - γ_tilde * V2(T)

H_0 = exact_human_capital_lemma_A7(M, dt, r_0, π_0)

println("\n--- Wealth Variation & % CE Analysis ---")
# Adjusted header for wider columns to fit the nominal calculation
println(rpad("F_0", 8), rpad("H_0/F_0", 10), rpad("W_0", 10), rpad("CE (Abs)", 12), rpad("Total % CE", 12), rpad("Real CE Gain", 15), "Nominal CE Gain")
println("-"^85)

# Test different starting financial wealths
F_test_values = [1.0, 10.0, 50.0, 100.0, 140.0, 300.0, 1000.0]

for F_val in F_test_values
    W_val = F_val + H_0
    CE_val = W_val * exp(exponent)

    # Total percentage CE
    CE_pct = (CE_val / W_val) * 100

    # Annualized Real CE gain (CAGR)
    annual_real_CE_gain = (CE_val / W_val)^(1.0 / T) - 1.0

    # Annualized Nominal CE gain using continuous expected inflation (exp(π) - 1)
    annual_nom_CE_gain = (1.0 + annual_real_CE_gain) * exp(overline_π) - 1.0

    # Calculate Human Capital to Financial Wealth ratio
    HC_to_F_ratio = H_0 / F_val

    println(rpad(round(F_val, digits=1), 8),
            rpad(round(HC_to_F_ratio, digits=3), 10),
            rpad(round(W_val, digits=1), 10),
            rpad(round(CE_val, digits=1), 12),
            rpad("$(round(CE_pct, digits=2))%", 12),
            rpad("$(round(annual_real_CE_gain * 100, digits=2))%", 15),
            "$(round(annual_nom_CE_gain * 100, digits=2))%")
end
println("-"^85)

# Set baseline values for the rest of the script's plotting functions
F_0 = 140.0
W_0 = F_0 + H_0
analytical_CE = W_0 * exp(exponent)
analytical_V0 = (W_0^(1 - γ)) / (1 - γ) * exp((γ - 1) * (-exponent))

annual_real_CE_gain_base = (analytical_CE / W_0)^(1.0 / T) - 1.0
annual_nom_CE_gain_base = (1.0 + annual_real_CE_gain_base) * exp(overline_π) - 1.0

println("\nBaseline Set For Plots (F_0 = 140.0):")
println("  Exact Human Capital (H_0): ", round(H_0, digits=4))
println("  Initial Total Wealth (W_0): ", round(W_0, digits=4))
println("  Analytical Expected Utility: ", round(analytical_V0, digits=6))
println("  Analytical CE:               ", round(analytical_CE, digits=4))
println("  Annualized Real CE Gain:     ", round(annual_real_CE_gain_base * 100, digits=2), "% per year")
println("  Annualized Nominal CE Gain:  ", round(annual_nom_CE_gain_base * 100, digits=2), "% per year")