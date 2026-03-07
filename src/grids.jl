
"""
    generate_adaptive_grid(u::Function, W_min::Float64, W_max::Float64, G_w::Int)

Automatically generates a wealth grid optimized for linear interpolation
based on the curvature (second derivative) of ANY user-provided utility function `u`.
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
    generate_linear_grid(min_val, max_val, N)

Generates a standard uniformly spaced grid between `min_val` and `max_val` with `N` points.
Useful for state variables (Z) or log-wealth (X).
"""
function generate_linear_grid(min_val::Float64, max_val::Float64, N::Int)
    return collect(range(min_val, max_val, length=N))
end

"""
    generate_log_spaced_grid(W_min, W_max, N)

Generates a grid where the points are uniformly spaced in log-space, but returned in levels.
This naturally clusters points near `W_min`.
"""
function generate_log_spaced_grid(W_min::Float64, W_max::Float64, N::Int)
    if W_min <= 0.0
        error("W_min must be strictly greater than 0 for a log-spaced grid.")
    end
    return exp.(collect(range(log(W_min), log(W_max), length=N)))
end
