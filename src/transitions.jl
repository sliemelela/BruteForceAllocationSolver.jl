"""
    make_merton_transition(r, μ, σ, dt)

Creates a transition function for the standard Merton model (Geometric Brownian Motion).
Returns a function with the signature `(Z, ε) -> (Z_next, R_e, R_base)` expected by the solver.
"""
function make_merton_transition(r::Float64, μ::Float64, σ::Float64, dt::Float64)
    # Precompute the risk-free rate since it's constant
    Rf = exp(r * dt)

    # Return the closure that the Bellman objective will call
    return function(Z::Vector{Float64}, ε::Vector{Float64})
        # ε[1] is the standard normal shock for the single risky asset
        Re = [exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * ε[1]) - Rf]

        # Returns: Z_next (empty), R^e (excess returns), R_base (risk-free rate)
        return Float64[], Re, Rf
    end
end