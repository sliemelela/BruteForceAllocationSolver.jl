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
function plot_mean_with_bounds(sim_data::Matrix{Float64}; title="Strategy over Time", ylabel="Value", color=:blue)
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
function plot_shock_comparison(baseline_sim::Matrix{Float64}, shocked_sim::Matrix{Float64};
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