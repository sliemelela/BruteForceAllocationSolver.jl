
"""
    generate_adaptive_grid(u::Function, W_min::Float64, W_max::Float64, G_w::Int)

Automatically generates a 1D grid optimized for linear interpolation by adapting to the
curvature (second derivative) of a user-provided utility function `u`.

The algorithm applies the principle of equidistribution. It evaluates a monitor function
``M(W) = √(|u''(W)| + ϵ)`` using automatic differentiation, and distributes grid points
such that the "curvature mass" is divided exactly evenly across all grid intervals.
This heavily clusters points in highly curved regions while spacing them out in flatter regions.

# Arguments
- `u::Function`: The pure utility function ``u(W)``. Must be twice-differentiable.
- `W_min::Float64`: The absolute minimum boundary of the grid.
- `W_max::Float64`: The absolute maximum boundary of the grid.
- `G_w::Int`: The total number of grid points to generate.

# Returns
- `Vector{Float64}`: An array of length `G_w` containing the optimized grid points.
"""
function generate_adaptive_grid(u::Function, W_min::Float64, W_max::Float64, G_w::Int)

    # Define the second derivative using ForwardDiff
    u_double_prime(W) = ForwardDiff.derivative(w -> ForwardDiff.derivative(u, w), W)

    # 2. Define the Monitor Function: M(W) = sqrt(|u''(W)| + ε)
    # ε prevents the density from dropping to zero in flat linear regions
    ε = 1e-5
    monitor(W, p) = sqrt(abs(u_double_prime(W)) + ε)

    # 3. Calculate the total "curvature mass" over the whole domain
    prob = IntegralProblem(monitor, W_min, W_max)
    total_mass = solve(prob, QuadGKJL()).u

    # 4. Find the grid points that divide the mass into equal chunks
    grid = zeros(Float64, G_w)
    grid[1] = W_min
    grid[end] = W_max

    target_fractions = range(0.0, 1.0, length=G_w)

    # A simple search to find the W that matches each cumulative fraction
    # (In a production package, you'd use Roots.jl to find these exactly)
    search_space = range(W_min, W_max, length=5000)
    cumulative_mass = cumsum([monitor(w, nothing) * step(search_space) for w in search_space])
    cumulative_mass ./= cumulative_mass[end] # Normalize to [0, 1]

    for i in 2:(G_w-1)
        # Find the index in our search space closest to the target fraction
        idx = findfirst(x -> x >= target_fractions[i], cumulative_mass)
        grid[i] = search_space[idx]
    end

    return grid
end


"""
    generate_linear_grid(min_val::Float64, max_val::Float64, N::Int)

Generates a standard uniformly spaced 1D array.

This grid is strictly linear. It is typically used for state variables (``Z``),
control spaces (like consumption fractions or portfolio weights), or problems formulated
in log-wealth space (``X = \\log(W)``).

# Arguments
- `min_val::Float64`: The starting value of the grid.
- `max_val::Float64`: The ending value of the grid.
- `N::Int`: The total number of evenly spaced points.

# Returns
- `Vector{Float64}`: A linear grid array of length `N`.
"""
function generate_linear_grid(min_val::Float64, max_val::Float64, N::Int)
    return collect(range(min_val, max_val, length=N))
end

"""
    generate_log_spaced_grid(W_min::Float64, W_max::Float64, N::Int)

Generates a 1D grid where points are uniformly spaced in logarithmic space,
but returned in absolute levels.

This naturally clusters points tightly near `W_min` and spreads them out as wealth
increases. This is the standard for problems involving CRRA utility
evaluated in absolute wealth levels, as it naturally accommodates the steep curvature near zero.

# Arguments
- `W_min::Float64`: The minimum boundary of the wealth grid. Must be strictly greater than `0.0`.
- `W_max::Float64`: The maximum boundary of the wealth grid.
- `N::Int`: The total number of grid points.

# Returns
- `Vector{Float64}`: An exponentially scaled grid array of length `N`.

# Throws
- `ErrorException`: If `W_min` is less than or equal to `0.0` (as the logarithm of zero is undefined).
"""
function generate_log_spaced_grid(W_min::Float64, W_max::Float64, N::Int)
    if W_min <= 0.0
        error("W_min must be strictly greater than 0 for a log-spaced grid.")
    end
    return exp.(collect(range(log(W_min), log(W_max), length=N)))
end
