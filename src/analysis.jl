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
    interp_c = [linear_interpolation((W_grid, Z_grids...), selectdim(pol_c, ndims(pol_c), t), extrapolation_bc=Interpolations.Flat()) for t in 1:M]

    # 2. Interpolate Portfolio Weights
    N_assets = length(pol_w[1])
    interp_w = []

    for t in 1:M
        asset_interps = []
        for a in 1:N_assets
            w_slice = map(x -> x[a], selectdim(pol_w, ndims(pol_w), t))
            push!(asset_interps, linear_interpolation((W_grid, Z_grids...), w_slice, extrapolation_bc=Interpolations.Flat()))
        end
        push!(interp_w, asset_interps)
    end

    return interp_c, interp_w
end

# src/analysis.jl (Replace the old certainty equivalent functions with this)

"""
    calculate_equivalent_consumption_stream(
        CE_array, u::Function, inv_u::Function, β::Float64, periods_remaining::Int
    )

Calculates the Constant Equivalent Consumption (CEC) stream.
This converts the lump-sum Certainty Equivalent (output by the solver) into a
guaranteed, constant consumption amount received every period that provides
the exact same lifetime utility.

# Arguments
- `CE_array`: An array or scalar of Certainty Equivalents (the solver's output grid).
- `u::Function`: The pure utility function.
- `inv_u::Function`: The mathematical inverse of the utility function.
- `β::Float64`: The subjective discount factor.
- `periods_remaining::Int`: The number of periods left until the terminal date.
"""
function calculate_equivalent_consumption_stream(
    CE_array::AbstractArray, u::Function, inv_u::Function, β::Float64, periods_remaining::Int
)
    # Sum of discount factors: 1 + β + β^2 + ... + β^periods_remaining
    discount_sum = sum(β^t for t in 0:periods_remaining)

    # Convert the CE back to lifetime utility, average it, and invert to per-period dollars
    return inv_u.( u.(CE_array) ./ discount_sum )
end

function calculate_equivalent_consumption_stream(
    CE::Float64, u::Function, inv_u::Function, β::Float64, periods_remaining::Int
)
    discount_sum = sum(β^t for t in 0:periods_remaining)
    return inv_u( u(CE) / discount_sum )
end