"""
    fractional_consumption(W, c)

Used when the state is raw Wealth, and the control `c` is a fraction (0 to 1).
"""
function fractional_consumption(W::Float64, c::Float64)
    return c * W
end

"""
    log_fractional_consumption(X, c)

Used when the state is Log-Wealth (X), and the control `c` is a fraction (0 to 1).
"""
function log_fractional_consumption(X::Float64, c::Float64)
    return c * exp(X)
end

"""
    absolute_consumption(state, c)

Used when the control `c` is an absolute dollar amount, rather than a fraction.
(The state is ignored because c is already the actual consumption).
"""
function absolute_consumption(state::Float64, c::Float64)
    return c
end

"""
    standard_budget_constraint(W, c, ω, R_e, R_base)

The standard multiplicative budget constraint for wealth evolution.
"""
function standard_budget_constraint(W, c, ω, R_e, R_base)
    return (1.0 - c) * W * (dot(ω, R_e) + R_base)
end

"""
    log_budget_constraint(X, c, ω, R_e, R_base)

The budget constraint for wealth evolution in log-space.
Here, the state variable `X` represents log(W).
"""
function log_budget_constraint(X, c, ω, R_e, R_base)
    # X_next = X + log(1 - c) + log(Portfolio Return)
    # We use max(..., 1e-10) to prevent log(negative) if extreme shocks occur
    port_return = dot(ω, R_e) + R_base

    if port_return <= 0.0
        return -Inf
    end

    return X + log(1.0 - c) + log(port_return)
end

"""
    make_crra_extrapolator(W_min, W_max, γ)

Creates an extrapolation strategy perfectly scaled for CRRA utility functions.
"""
function make_crra_extrapolator(W_min::Float64, W_max::Float64, γ::Float64)
    return function(W_next::Float64, Z_next::Vector{Float64}, V_next_interp)
        if W_next < W_min
            W_next = max(W_next, 1e-10) # Prevent log(0)
            return V_next_interp(W_min, Z_next...) * (W_next / W_min)^(1.0 - γ)
        elseif W_next > W_max
            return V_next_interp(W_max, Z_next...) * (W_next / W_max)^(1.0 - γ)
        else
            return V_next_interp(W_next, Z_next...)
        end
    end
end

"""
    make_log_crra_extrapolator(X_min, X_max, γ)

Creates an extrapolation strategy perfectly scaled for CRRA utility
when the state variable is log(Wealth).
"""
function make_log_crra_extrapolator(X_min::Float64, X_max::Float64, γ::Float64)
    return function(X_next::Float64, Z_next::Vector{Float64}, V_next_interp)
        if X_next < X_min
            # In log space, (W / W_min)^(1-γ) becomes exp((1-γ) * (X - X_min))
            return V_next_interp(X_min, Z_next...) * exp((1.0 - γ) * (X_next - X_min))
        elseif X_next > X_max
            return V_next_interp(X_max, Z_next...) * exp((1.0 - γ) * (X_next - X_max))
        else
            return V_next_interp(X_next, Z_next...)
        end
    end
end