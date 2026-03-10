using BruteForceAllocationSolver
using FastGaussQuadrature
using Test

@testset "Analysis and Plotting Module" begin
    # 1. Setup Mock Grids and Policy Arrays
    W_grid = [1.0, 2.0, 3.0]
    Z_grids = [[0.1, 0.2]] # One auxiliary state variable
    M = 10 # 10 timesteps

    # Create mock policy arrays
    # Shape: (length(W_grid), length(Z_grids[1]), M) -> (3, 2, 10)
    pol_c = zeros(3, 2, 10)
    for t in 1:10
        pol_c[:, :, t] .= t * 0.10 # Consume more over time
    end

    # Portfolio arrays contain Vectors (e.g., 1 risky asset)
    pol_w = fill([0.0], 3, 2, 10)
    for t in 1:10
        pol_w[:, :, t] .= [[0.50 * (11 - t) / 10]] # Less risk over time
    end

    # 2. Test Policy Interpolators
    interp_c, interp_w = create_policy_interpolators(pol_c, pol_w, W_grid, Z_grids)

    # 3. Create mock Monte Carlo simulation data (100 paths x 10 timesteps)
    # We add some random noise to make the plots look realistic
    sim_base = zeros(100, M)
    sim_shock = zeros(100, M)
    for path in 1:100, t in 1:M
        sim_base[path, t] = 0.5 + 0.05 * randn()
        # The shocked scenario drops at time 5
        sim_shock[path, t] = t < 5 ? sim_base[path, t] : 0.3 + 0.05 * randn()
    end

    # 4. Generate the Figures
    fig1 = plot_mean_with_bounds(sim_base, title="Mock Strategy over Time", ylabel="Fraction")
    fig2 = plot_shock_comparison(sim_base, sim_shock, shock_time=5, title="Mock Shock Impact", ylabel="Fraction")
    fig3 = plot_policy_vs_state(pol_c, W_grid, Z_grids, 5, plot_against_W=true, ylabel="Consumption")

    # 5. Assertions
    @test fig1 isa CairoMakie.Figure
    @test fig2 isa CairoMakie.Figure
    @test fig3 isa CairoMakie.Figure

    # 6. SAVE THE PLOTS!
    # This will generate 3 PNG files in your project directory so you can actually look at them
    save("test_plot_mean_bounds.png", fig1)
    save("test_plot_shock.png", fig2)
    save("test_plot_policy_state.png", fig3)

    println("Saved 3 test plots to your active directory!")
end

using BruteForceAllocationSolver
using FastGaussQuadrature
using CairoMakie
using Test

# ==========================================
# 1. Setup & Solve the Standard Merton Model
# ==========================================
println("Solving Standard Merton Model...")

M = 10; dt = 1.0; β = 0.96; γ = 5.0
u(x) = (x^(1 - γ))/(1 - γ)

G_w, W_min, W_max = 500, 1.0, 100.0
W_grid = generate_log_spaced_grid(W_min, W_max, G_w)
Z_grids = Vector{Float64}[]
c_grid = generate_linear_grid(0.01, 0.99, 50)
omega_space = [[ω] for ω in generate_linear_grid(0.0, 1.0, 101)]

Q = 10
ε_nodes, W_weights = gausshermite(Q, normalize=true)
ε_nodes = [[n] for n in ε_nodes]

r = 0.02; μ = 0.07; σ = 0.20
merton_transition = make_merton_transition(r, μ, σ, dt)
crra_extrapolator = make_crra_extrapolator(W_grid[1], W_grid[end], γ)

V, pol_c, pol_w = solve_dynamic_program(
    W_grid, Z_grids, c_grid, omega_space,
    ε_nodes, W_weights, merton_transition,
    M, β, u, fractional_consumption,
    standard_budget_constraint, crra_extrapolator
)

# Extract scalar weights from the portfolio vectors for plotting
pol_w_scalar = map(x -> x[1], pol_w)

# ==========================================
# 2. Generate Static Policy Plots
# ==========================================
println("Generating Policy Plots...")

# Plot 1: Portfolio Weight vs Wealth (Should be a perfectly flat line at 0.25!)
fig_w_policy = plot_policy_vs_state(pol_w_scalar, W_grid, Z_grids, 1,
                                    plot_against_W=true, ylabel="Portfolio Weight ω")

# Plot 2: Consumption vs Wealth across different timesteps
fig_c_policy = Figure(size = (800, 400))
ax_c = Axis(fig_c_policy[1, 1], title="Consumption Fraction vs Wealth over Time",
            xlabel="Wealth (W)", ylabel="Consumption Fraction c")

# Plot the consumption policy at time t=1, t=5, and t=9
lines!(ax_c, W_grid, pol_c[:, 1], linewidth=3, label="Time 1", color=:dodgerblue)
lines!(ax_c, W_grid, pol_c[:, 5], linewidth=3, label="Time 5", color=:darkorange)
lines!(ax_c, W_grid, pol_c[:, 9], linewidth=3, label="Time 9", color=:firebrick)
axislegend(ax_c, position=:lt)

# ==========================================
# 3. Forward Simulation
# ==========================================
println("Simulating Forward Paths...")

# Create continuous interpolators from the solved grids
interp_c, interp_w = create_policy_interpolators(pol_c, pol_w, W_grid, Z_grids)

N_paths = 1000
W_sim = zeros(N_paths, M)
c_sim = zeros(N_paths, M)
w_sim = zeros(N_paths, M)

# We use the same transition function to simulate the actual market!
Rf = exp(r * dt)

for path in 1:N_paths
    W_t = 50.0 # Start with $50

    for t in 1:M
        # 1. Agent makes choices using the continuous interpolator
        c_t = interp_c[t](W_t)
        ω_t = interp_w[t][1](W_t) # Asset 1

        c_sim[path, t] = c_t
        w_sim[path, t] = ω_t

        # 2. Market shock occurs
        ε_realized = randn()
        Re_realized = exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * ε_realized) - Rf

        # 3. Wealth evolves
        W_t = (1.0 - c_t) * W_t * (ω_t * Re_realized + Rf)
        W_sim[path, t] = W_t
    end
end

# ==========================================
# 4. Generate Simulation Plots
# ==========================================
# Plot 3: The mean trajectory of wealth over time with 10th-90th percentile bounds
fig_w_sim = plot_mean_with_bounds(W_sim, title="Simulated Wealth over Time", ylabel="Wealth", color=:seagreen)

# Plot 4: The mean trajectory of consumption fraction
fig_c_sim = plot_mean_with_bounds(c_sim, title="Simulated Consumption Fraction", ylabel="Fraction of Wealth", color=:purple)

# ==========================================
# 5. Display / Save All Plots
# ==========================================
# Displaying them interactively
display(fig_w_policy)
display(fig_c_policy)
display(fig_w_sim)
display(fig_c_sim)

# Save to disk
save("merton_portfolio_policy.png", fig_w_policy)
save("merton_consumption_policy.png", fig_c_policy)
save("merton_simulated_wealth.png", fig_w_sim)
save("merton_simulated_consumption.png", fig_c_sim)
println("Plots successfully generated and saved!")