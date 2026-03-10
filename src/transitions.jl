"""
    make_merton_transition(r::Float64, μ::Float64, σ::Float64, dt::Float64)

Creates a market transition strategy for the classic Samuelson-Merton portfolio choice model.

This factory function models a financial market with one risk-free asset and one risky asset
following Geometric Brownian Motion (GBM). It precomputes the constant risk-free rate to maximize
performance and returns a highly optimized closure that evaluates the stochastic returns.

Mathematically, the realized gross risk-free return is ``R_f = \\exp(r \\cdot dt)``, and the
gross return of the risky asset is ``R_{risky} = \\exp((μ - 1/2 σ^2)dt + σ√(dt) ε)``.

# Arguments
- `r::Float64`: The continuously compounded risk-free interest rate.
- `μ::Float64`: The expected continuous return (drift) of the risky asset.
- `σ::Float64`: The volatility of the risky asset.
- `dt::Float64`: The time step size (``δt``).

# Returns
- `Function`: A closure with the signature `(Z::Vector{Float64}, ε::Vector{Float64}) -> Tuple`.
  When called by the solver, this closure returns:
  1. `Z_next`: An empty `Float64[]` array, as the standard Merton model has no auxiliary state variables.
  2. `R_e`: A 1-element vector containing the excess return of the risky asset over the risk-free rate (``R_{risky} - R_f``).
  3. `R_base`: The deterministic gross risk-free return (``R_f``).
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