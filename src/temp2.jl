using LinearAlgebra
using ForwardDiff
using Interpolations
using Integrals
using FastGaussQuadrature
using Plots

function evaluate_bellman_objective(
    W_n::Float64,
    Z_n::Vector{Float64},
    c_n::Float64,
    ω_n::Vector{Float64},
    ε_nodes::Vector{Vector{Float64}},
    W_weights::Vector{Float64},
    V_next,                     # Callable interpolation object
    transition_model::Function, # The model-specific dynamics
    β::Float64,
    u::Function,                # Utility function
    W_min::Float64,
)

    expected_future_value = 0.0

    # Loop over all precomputed quadrature nodes and weights
    for j in 1:length(ε_nodes)
        ε_j = ε_nodes[j]
        weight = W_weights[j]

        # Calculate Z_next, R^e, and R
        Z_next, R_e, R_base = transition_model(Z_n, ε_j)

        # Calculate W_next
        W_next = (1.0 - c_n) * W_n * (dot(ω_n, R_e) + R_base)

        # Guardrail: Prevent negative wealth crashes going into the utility function
        if W_next < W_min
            # Prevent log(0) domain errors by clamping to a microscopic number
            W_next = max(W_next, 1e-10)

            # The future value is heavily punished by the true curvature of u()
            expected_future_value += weight * u(W_next)
        else
            expected_future_value += weight * V_next(W_next, Z_next...)
        end
    end

    # Calculate current utility
    # Guardrail: CRRA utility evaluates to -Inf if consumption is zero
    if c_n > 0.0
        current_utility = u(c_n * W_n)
    else
        current_utility = -Inf
    end

    # Return the total objective value for this specific (c_n, ω_n) choice
    return current_utility + β * expected_future_value
end

function optimize_controls_brute_force(
    W_n::Float64,
    Z_n::Vector{Float64},
    c_grid::Vector{Float64},
    omega_space,                      # An iterable of ω vectors
    ε_nodes::Vector{Vector{Float64}},
    W_weights::Vector{Float64},
    V_next,                           # Callable interpolation object
    transition_model::Function,       # The model-specific dynamics
    β::Float64,
    u::Function,                      # Utility function
    W_min::Float64
)

    # Initialize trackers for the maximum value and the optimal policies
    best_val = -Inf
    best_c = 0.0
    best_ω = first(omega_space) # Fallback to the first valid vector

    for c_n in c_grid, ω_n in omega_space

        current_val = evaluate_bellman_objective(
            W_n, Z_n, c_n, ω_n,
            ε_nodes, W_weights,
            V_next, transition_model, β, u, W_min
        )

        if current_val > best_val
            best_val = current_val
            best_c = c_n
            best_ω = ω_n
        end
    end

    return best_val, best_c, best_ω
end



function solve_dynamic_program(
    W_grid::Vector{Float64},
    Z_grids::Vector{Vector{Float64}}, # A list of grids for the state variables
    c_grid::Vector{Float64},
    omega_space::Vector{Vector{Float64}},
    ε_nodes::Vector{Vector{Float64}},
    W_weights::Vector{Float64},
    transition_model::Function,
    M::Int,
    β::Float64,
    u::Function
)

    # Determine the exact dimensions of our state space
    sz = (length(W_grid), (length(z) for z in Z_grids)...)

    # Allocate storage arrays
    # V has an extra dimension for time, spanning from 1 to M+1
    V     = zeros(Float64, sz..., M + 1)
    pol_c = zeros(Float64, sz..., M)
    pol_w = Array{Vector{Float64}}(undef, sz..., M) # Array of vectors for portfolio weights

    # Find minimum W
    W_min = W_grid[1]

    # Terminal Condition (Time step M+1)
    println("Setting terminal conditions...")
    for idx in CartesianIndices(sz)
        W_terminal = W_grid[idx[1]]
        V[idx, M+1] = u(W_terminal)
    end

    # The Backwards Recursion Loop
    println("Starting backwards recursion from step $M down to 1...")
    for n in M:-1:1
        println("  Solving timestep: $n")

        # Extract the value function from time n+1 (using a memory-efficient view)
        V_next_data = selectdim(V, ndims(V), n + 1)

        # Build the continuous multidimensional interpolation object dynamically
        V_next_interp = linear_interpolation(
            (W_grid, Z_grids...),
            V_next_data,
            extrapolation_bc=Line()
        )

        # Iterate over every point in the current state space
        Threads.@threads for idx in CartesianIndices(sz)

            # Extract current state values based on the multidimensional index
            W_n = W_grid[idx[1]]

            # Dynamically extract the Z states
            Z_n = [Z_grids[k][idx[1+k]] for k in 1:length(Z_grids)]

            best_val, best_c, best_ω = optimize_controls_brute_force(
                W_n, Z_n, c_grid, omega_space,
                ε_nodes, W_weights,
                V_next_interp, transition_model, β, u, W_min
            )

            # Store the optimal policy and value
            V[idx, n]     = best_val
            pol_c[idx, n] = best_c
            pol_w[idx, n] = best_ω
        end
    end

    println("Recursion complete.")
    return V, pol_c, pol_w
end


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



function run()

    # Utility function
    γ = 5.0
    u(x) = (x^(1 - γ))/(1 - γ)

    # Grids
    G_w = 100
    W_min = 1.0
    W_max = 100.0
    # W_grid = generate_adaptive_grid(u, W_min, W_max, G_w)
    W_grid = exp.(collect(range(log(W_min), log(W_max), length=G_w)))
    Z_grids = Vector{Float64}[] # No state variables!

    c_grid = collect(range(0.01, 0.99, length=50))
    w_grid_1D = collect(range(0.0, 1.0, length=101))
    omega_space = [[w] for w in w_grid_1D]

    # Quadrature (1D for Merton) #TODO CHECK
    Q = 10
    nodes, weights = gausshermite(Q)
    # display(nodes)
    ε_nodes = [[n * sqrt(2.0)] for n in nodes]
    W_weights = weights ./ sqrt(pi)


    # display(ε_nodes)
    # display(W_weights)

    # Model Dynamics Function
    function merton_transition(Z::Vector{Float64}, ε::Vector{Float64})
        # Z is empty, ε has 1 element
        r = 0.02; μ = 0.07; σ = 0.20; dt = 1.0

        Rf = exp(r * dt)
        Re = [exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * ε[1]) - Rf]

        # Return: Z_next, R^e, R_base
        return Float64[], Re, Rf
    end

    # Run the solver!
    V, pol_c, pol_w = solve_dynamic_program(
        W_grid, Z_grids, c_grid, omega_space,
        ε_nodes, W_weights, merton_transition,
        10, 0.96, u
    )
    return pol_w
end
pol_w = run()
nothing
display(pol_w)


# # ==========================================
# # 2. Define the Economic Model
# # ==========================================
# γ = 5.0
# u(W) = (W^(1 - γ)) / (1 - γ)

# W_min = 0.5
# W_max = 100.0
# G_w = 25 # Using a small number of points so they are visually distinct

# # Generate both grids for comparison
# adaptive_grid = generate_adaptive_grid(u, W_min, W_max, G_w)
# linear_grid = collect(range(W_min, W_max, length=G_w))

# # ==========================================
# # 3. Visual Verification
# # ==========================================
# # Plot the underlying utility function smoothly
# W_plot = range(W_min, W_max, length=500)
# u_plot = u.(W_plot)

# plt = plot(W_plot, u_plot, label="True CRRA Utility (γ=5)",
#            linewidth=2, color=:black, legend=:bottomright,
#            title="Curvature-Adaptive Grid vs. Linear Grid",
#            xlabel="Wealth (W)", ylabel="Utility u(W)")

# # Plot the Linear Grid points on the curve
# scatter!(plt, linear_grid, u.(linear_grid),
#          label="Linear Grid (25 pts)", markersize=5,
#          color=:red, marker=:x)

# # Plot the Adaptive Grid points on the curve
# scatter!(plt, adaptive_grid, u.(adaptive_grid),
#          label="Adaptive Grid (25 pts)", markersize=5,
#          color=:blue, marker=:circle)

# display(plt)

# # Print the first 10 points of the adaptive grid to the console
# println("First 10 points of the Adaptive Grid:")
# display(round.(adaptive_grid[1:10], digits=3))
