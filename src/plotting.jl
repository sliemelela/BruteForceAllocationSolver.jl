using CairoMakie
using Statistics

"""
    plot_heatmap(x_grid, y_grid, data_2d; ...)

A fully agnostic 2D heatmap contour plot.
Pass in any two grids (e.g., Wealth vs Interest Rate, or Interest Rate vs Inflation)
and a 2D matrix of corresponding data (Value, CE, or Policy weights).
"""
function plot_heatmap(x_grid::AbstractVector, y_grid::AbstractVector, data_2d::AbstractMatrix;
                      title="Heatmap", xlabel="X", ylabel="Y",
                      colormap=:viridis, label="Value")
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    co = contourf!(ax, x_grid, y_grid, data_2d, colormap=colormap, levels=20)
    Colorbar(fig[1, 2], co, label=label)

    return fig
end

"""
    plot_curves(x_grid, data_series::Vector, labels::Vector{String}; ...)

A fully agnostic line plotter. Pass an X-axis and an array of Y-data vectors.
Perfect for Value Functions across time, CE progression, or policy rules.
"""
function plot_curves(x_grid::AbstractVector, data_series::Vector, labels::Vector{String};
                     title="Curves", xlabel="X", ylabel="Y", legend_pos=:rt)
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    # Automatically generate colors for however many series are passed
    colors = cgrad(:tab10, length(data_series), categorical=true)

    for (i, y_data) in enumerate(data_series)
        lines!(ax, x_grid, y_data, linewidth=3, color=colors[i], label=labels[i])
    end

    axislegend(ax, position=legend_pos)

    return fig
end

function plot_mean_with_bounds(sim_data::AbstractMatrix{Float64}; title="Strategy over Time", ylabel="Value", color=:blue)
    M = size(sim_data, 2)
    times = 1:M
    means = mean(sim_data, dims=1)[:]
    p10 = [quantile(sim_data[:, t], 0.10) for t in times]
    p90 = [quantile(sim_data[:, t], 0.90) for t in times]

    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel="Time (Steps)", ylabel=ylabel)
    band!(ax, times, p10, p90, color=(color, 0.2), label="10th-90th Percentile")
    lines!(ax, times, means, color=color, linewidth=3, label="Mean")
    axislegend(ax, position=:rt)
    return fig
end

function plot_paths_overlay(sim_data::AbstractMatrix{Float64};
                            num_paths=20, title="Simulated Paths Overlay",
                            xlabel="Time (Steps)", ylabel="Value",
                            line_color=:dodgerblue, mean_color=:black,
                            band_color=(:black, 0.1))
    sims, M = size(sim_data)
    times = 1:M
    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], title=title, xlabel=xlabel, ylabel=ylabel)

    mean_path = vec(mean(sim_data, dims=1))
    p10 = [quantile(sim_data[:, t], 0.10) for t in times]
    p90 = [quantile(sim_data[:, t], 0.90) for t in times]

    band!(ax, times, p10, p90, color=band_color, label="10th-90th Percentile")

    n_plot = min(sims, num_paths)
    sampled_indices = rand(1:sims, n_plot)

    for (idx, i) in enumerate(sampled_indices)
        lbl = idx == 1 ? "Sampled Paths" : nothing
        lines!(ax, times, sim_data[i, :], color=(line_color, 0.15), linewidth=1.5, label=lbl)
    end

    lines!(ax, times, mean_path, color=mean_color, linewidth=3, label="Mean")
    axislegend(ax, position=:rt)
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