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
    calculate_certainty_equivalent(V_array, inv_u::Function)

Calculates the Certainty Equivalent (CE) Wealth for an array of expected
utilities (Value Function), entirely agnostic to the utility function used.
Used primarily for pure Terminal Wealth problems.

# Arguments
- `V_array`: An array or scalar of expected utility values.
- `inv_u::Function`: The mathematical inverse of the utility function used
  during the dynamic programming solution.
"""
function calculate_certainty_equivalent(V_array::AbstractArray, inv_u::Function)
    # The dot (.) syntax efficiently broadcasts the function across the array
    return inv_u.(V_array)
end

# Dispatch for a single scalar value
function calculate_certainty_equivalent(V::Float64, inv_u::Function)
    return inv_u(V)
end

"""
    calculate_equivalent_consumption_stream(
        V_array, inv_u::Function, β::Float64, periods_remaining::Int
    )

Calculates the Constant Equivalent Consumption (CEC) stream.
This is the guaranteed, constant consumption amount received every period
that provides the same total lifetime utility as the dynamic optimal policy.
Used for Intermediate Consumption problems.

# Arguments
- `V_array`: An array or scalar of expected utility values.
- `inv_u::Function`: The mathematical inverse of the utility function.
- `β::Float64`: The subjective discount factor used in the solver.
- `periods_remaining::Int`: The number of periods left until the terminal date.
"""
function calculate_equivalent_consumption_stream(
    V_array::AbstractArray, inv_u::Function, β::Float64, periods_remaining::Int
)
    # Sum of discount factors: 1 + β + β^2 + ... + β^periods_remaining
    discount_sum = sum(β^t for t in 0:periods_remaining)

    # Normalize the total value by the discount sum, then invert the utility
    return inv_u.(V_array ./ discount_sum)
end

function calculate_equivalent_consumption_stream(
    V::Float64, inv_u::Function, β::Float64, periods_remaining::Int
)
    discount_sum = sum(β^t for t in 0:periods_remaining)
    return inv_u(V / discount_sum)
end