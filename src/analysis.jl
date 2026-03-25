"""
    create_policy_interpolators(pol_c, pol_w, W_grid, Z_grids)

Converts the discrete policy grids into continuous interpolation functions
so they can be evaluated along arbitrary simulated paths.
"""
function create_policy_interpolators(pol_c, pol_w, W_grid, Z_grids)
    M = size(pol_c, ndims(pol_c))

    # 1. Interpolate Consumption
    # We use Flat() extrapolation so if wealth goes off-grid during simulation,
    # the agent just uses the boundary policy.
    interp_c = [linear_interpolation((W_grid, Z_grids...), selectdim(pol_c, ndims(pol_c), t), extrapolation_bc=Flat()) for t in 1:M]

    # 2. Interpolate Portfolio Weights
    N_assets = length(pol_w[1])
    interp_w = []

    for t in 1:M
        asset_interps = []
        for a in 1:N_assets
            w_slice = map(x -> x[a], selectdim(pol_w, ndims(pol_w), t))
            push!(asset_interps, linear_interpolation((W_grid, Z_grids...), w_slice, extrapolation_bc=Flat()))
        end
        push!(interp_w, asset_interps)
    end

    return interp_c, interp_w
end

"""
    plot_mean_with_bounds(sim_data::Matrix{Float64}; title="", ylabel="Value", color=:blue)

Creates a CairoMakie Figure showing the mean of a simulated variable (e.g., consumption
or portfolio weight) over time, surrounded by a shaded 10th-90th percentile band.
"""
function plot_mean_with_bounds(sim_data::AbstractMatrix{Float64}; title="Strategy over Time", ylabel="Value", color=:blue)
    M = size(sim_data, 2)
    times = 1:M

    # Calculate statistics across all simulated paths (dim 1)
    means = mean(sim_data, dims=1)[:]
    p10 = [quantile(sim_data[:, t], 0.10) for t in times]
    p90 = [quantile(sim_data[:, t], 0.90) for t in times]

    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel="Time (Steps)", ylabel=ylabel)

    # Add the confidence band and the mean line
    band!(ax, times, p10, p90, color=(color, 0.2), label="10th-90th Percentile")
    lines!(ax, times, means, color=color, linewidth=3, label="Mean")

    axislegend(ax, position=:rt) # Right-Top

    return fig
end

"""
    plot_shock_comparison(baseline_sim, shocked_sim; shock_time=nothing, title="", ylabel="")

Compares the mean trajectory of a baseline simulation against a shocked scenario.
"""
function plot_shock_comparison(baseline_sim::AbstractMatrix{Float64}, shocked_sim::AbstractMatrix{Float64};
                               shock_time=nothing, title="Shock Comparison", ylabel="Value")
    M = size(baseline_sim, 2)
    times = 1:M

    base_mean = mean(baseline_sim, dims=1)[:]
    shock_mean = mean(shocked_sim, dims=1)[:]

    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel="Time (Steps)", ylabel=ylabel)

    lines!(ax, times, base_mean, color=:blue, linewidth=3, label="Baseline (Mean)")
    lines!(ax, times, shock_mean, color=:red, linewidth=3, linestyle=:dash, label="Shocked Scenario")

    # Draw a vertical line where the shock occurred
    if shock_time !== nothing
        vlines!(ax, [shock_time], color=:black, linestyle=:dot, label="Shock Applied")
    end

    axislegend(ax, position=:rt)

    return fig
end

"""
    plot_policy_vs_state(pol_matrix, W_grid, Z_grids, target_time; plot_against_W=true, ...)

Plots the exact policy function at a specific timestep against one state variable,
holding all other state variables constant.
"""
function plot_policy_vs_state(pol_matrix, W_grid, Z_grids, target_time::Int;
                              plot_against_W=true, fixed_Z_idx=1, fixed_W_idx=1, ylabel="Policy Choice")

    fig = Figure(size = (800, 400))

    if plot_against_W
        # Vary Wealth, hold Z constant
        y_data = [pol_matrix[i, fixed_Z_idx, target_time] for i in 1:length(W_grid)]

        ax = Axis(fig[1, 1], title="Policy vs Wealth (t=$target_time)", xlabel="Wealth (W)", ylabel=ylabel)
        lines!(ax, W_grid, y_data, linewidth=3, color=:dodgerblue)
    else
        # Vary Z, hold Wealth constant (Assuming 1 auxiliary state variable for simplicity)
        y_data = [pol_matrix[fixed_W_idx, j, target_time] for j in 1:length(Z_grids[1])]

        ax = Axis(fig[1, 1], title="Policy vs State Z (t=$target_time)", xlabel="State Variable (Z)", ylabel=ylabel)
        lines!(ax, Z_grids[1], y_data, linewidth=3, color=:darkorange)
    end

    return fig
end

"""
    plot_policy_heatmap(pol_matrix, W_grid, Z_grids, target_time; ...)

Creates a 2D contour plot (heatmap) showing the optimal policy choice against
both Wealth (W) and an auxiliary state variable (Z) at a specific timestep.
"""
function plot_policy_heatmap(pol_matrix, W_grid, Z_grids, target_time::Int;
                             fixed_Z_idx=1, title="Policy Heatmap",
                             xlabel="Wealth (W)", ylabel="State Variable (Z)")

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    # Assumes at least one auxiliary Z grid exists
    Z_grid = Z_grids[1]

    # Extract the 2D slice for the chosen timestep
    slice_2d = [pol_matrix[w, z, target_time] for w in 1:length(W_grid),
                z in 1:length(Z_grid)]

    # Generate the filled contour
    co = contourf!(ax, W_grid, Z_grid, slice_2d, colormap=:viridis, levels=20)
    Colorbar(fig[1, 2], co, label="Policy Value")

    return fig
end

"""
    plot_value_function(V_matrix, W_grid, timesteps; ...)

Plots the value function curve against Wealth at specific timesteps.
Useful for verifying that boundary extrapolators stitch perfectly without kinks.
"""
function plot_value_function(V_matrix, W_grid, timesteps::Vector{Int};
                             fixed_Z_idx=1, title="Value Function V(W)",
                             xlabel="Wealth (W)", ylabel="Expected Utility")

    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    # Generate distinct colors for each timestep line
    colors = cgrad(:tab10, length(timesteps), categorical=true)

    for (i, t) in enumerate(timesteps)
        # Handle multi-dimensional V (assumes 1 auxiliary state for now)
        y_data = ndims(V_matrix) == 2 ?
                 V_matrix[:, t] : V_matrix[:, fixed_Z_idx, t]

        lines!(ax, W_grid, y_data, linewidth=3, color=colors[i], label="t = $t")
    end

    axislegend(ax, position=:rb) # Right-Bottom

    return fig
end

"""
    plot_paths_overlay(sim_data::Matrix{Float64}; ...)

Plots individual Monte Carlo paths overlaid on the mean trajectory and
confidence bounds to visualize the actual volatility and skewness.
"""
function plot_paths_overlay(sim_data::AbstractMatrix{Float64};
                            num_paths=20, title="Simulated Paths Overlay",
                            xlabel="Time (Steps)", ylabel="Value",
                            line_color=:dodgerblue, mean_color=:black,
                            band_color=(:black, 0.1))

    sims, M = size(sim_data)
    times = 1:M

    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    # Calculate statistics
    mean_path = vec(mean(sim_data, dims=1))
    p10 = [quantile(sim_data[:, t], 0.10) for t in times]
    p90 = [quantile(sim_data[:, t], 0.90) for t in times]

    # Plot the confidence band first (so it sits in the background)
    band!(ax, times, p10, p90, color=band_color, label="10th-90th Percentile")

    # Randomly sample paths to avoid overwhelming the plot
    n_plot = min(sims, num_paths)
    sampled_indices = rand(1:sims, n_plot)

    # Plot the individual "spaghetti" lines with high transparency
    for (idx, i) in enumerate(sampled_indices)
        # Only attach the label to the first line to keep the legend clean
        lbl = idx == 1 ? "Sampled Paths" : nothing
        lines!(ax, times, sim_data[i, :], color=(line_color, 0.15), linewidth=1.5, label=lbl)
    end

    # Plot the mean as a thick solid line on top
    lines!(ax, times, mean_path, color=mean_color, linewidth=3, label="Mean")

    axislegend(ax, position=:rt)

    return fig
end

"""
    plot_objective_curve(x_vals, y_vals; title="", xlabel="", ylabel="", color=:purple)

Plots a 2D cross-section of the Bellman objective function to verify that the
numerical grid captures the true parabolic maximum without clipping at the boundaries.
"""
function plot_objective_curve(x_vals::AbstractVector, y_vals::AbstractVector;
                              title="Objective Function", xlabel="Weight",
                              ylabel="Expected Utility", color=:purple)
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    # Filter out -Inf values so CairoMakie doesn't complain
    valid_idx = isfinite.(y_vals)
    clean_x = x_vals[valid_idx]
    clean_y = y_vals[valid_idx]

    lines!(ax, clean_x, clean_y, linewidth=3, color=color)

    if !isempty(clean_y)
        max_idx = argmax(clean_y)
        scatter!(ax, [clean_x[max_idx]], [clean_y[max_idx]],
                 color=:red, markersize=15, label="Numerical Maximum")
        axislegend(ax, position=:rt)
    end

    return fig
end

"""
    plot_deterministic_glidepath(times, wN_vals, wS_vals; ...)

Plots the exact optimal portfolio weights over time for a strictly fixed state
(e.g., average wealth, average interest rate, average inflation). This isolates
the pure life-cycle "glidepath" effect from the noise of simulated market shocks.
"""
function plot_deterministic_glidepath(times::AbstractVector, wN_vals::AbstractVector, wS_vals::AbstractVector;
                                      title="Deterministic Target-Date Glidepath",
                                      xlabel="Time (Steps)", ylabel="Portfolio Weight")
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    lines!(ax, times, wN_vals, linewidth=3, color=:blue, label="Nominal Bond Weight")
    lines!(ax, times, wS_vals, linewidth=3, color=:green, label="Stock Weight")

    axislegend(ax, position=:rt)
    return fig
end

"""
    plot_wealth_composition(times, W_vals, H_vals; ...)

Plots Financial Wealth (W) against the present value of Human Capital (H)
over time to contextualize portfolio shifts as human capital depletes.
"""
function plot_wealth_composition(times::AbstractVector, W_vals::AbstractVector, H_vals::AbstractVector;
                                 title="Wealth Composition Over Time", xlabel="Time (Steps)", ylabel="Value")
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    lines!(ax, times, W_vals, linewidth=3, color=:dodgerblue, label="Mean Financial Wealth (W)")
    lines!(ax, times, H_vals, linewidth=3, color=:darkorange, label="Human Capital (H)")

    axislegend(ax, position=:rc)
    return fig
end