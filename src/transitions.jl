# src/transitions.jl (Update the return statements inside the closures)

function make_merton_transition(r::Float64, μ::Float64, σ::Float64, dt::Float64)
    Rf = exp(r * dt)
    return function(Z, ε)
        Re = SVector(exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * ε[1]) - Rf)
        return SVector{0, Float64}(), Re, Rf
    end
end

function make_stochastic_r_constant_mu_transition(κ::Float64, θ::Float64, σ_r::Float64, μ::Float64, σ_S::Float64, ρ::Float64, dt::Float64)
    return function(Z, ε)
        r_n = Z[1]
        r_next = r_n + κ * (θ - r_n) * dt + σ_r * sqrt(dt) * ε[1]

        Rf = exp(r_n * dt)
        R_S = exp((μ - 0.5 * σ_S^2) * dt + σ_S * sqrt(dt) * ε[2])

        return SVector(r_next), SVector(R_S - Rf), Rf
    end
end

function make_stochastic_r_constant_premium_transition(κ::Float64, θ::Float64, σ_r::Float64, λ_S::Float64, σ_S::Float64, ρ::Float64, dt::Float64)
    return function(Z, ε)
        r_n = Z[1]
        r_next = r_n + κ * (θ - r_n) * dt + σ_r * sqrt(dt) * ε[1]

        Rf = exp(r_n * dt)
        R_S = exp((r_n + λ_S * σ_S - 0.5 * σ_S^2) * dt + σ_S * sqrt(dt) * ε[2])

        return SVector(r_next), SVector(R_S - Rf), Rf
    end
end

function make_stochastic_r_bond_stock_transition(κ::Float64, θ::Float64, σ_r::Float64, λ_r::Float64, τ::Float64, λ_S::Float64, σ_S::Float64, ρ::Float64, dt::Float64)
    B_r = abs(κ) < 1e-8 ? τ : (1.0 - exp(-κ * τ)) / κ
    bond_vol = -B_r * σ_r

    return function(Z, ε)
        r_n = Z[1]
        r_next = r_n + κ * (θ - r_n) * dt + σ_r * sqrt(dt) * ε[1]

        Rf = exp(r_n * dt)
        R_bond = exp((r_n - λ_r * B_r * σ_r - 0.5 * bond_vol^2) * dt + bond_vol * sqrt(dt) * ε[1])
        R_S = exp((r_n + λ_S * σ_S - 0.5 * σ_S^2) * dt + σ_S * sqrt(dt) * ε[2])

        return SVector(r_next), SVector(R_bond - Rf, R_S - Rf), Rf
    end
end