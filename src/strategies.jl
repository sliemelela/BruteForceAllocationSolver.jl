"""
    fractional_consumption(W, c)
"""
function fractional_consumption(W::Real, c::Real)
    return c * W
end

"""
    log_fractional_consumption(X, c)
"""
function log_fractional_consumption(X::Real, c::Real)
    return c * exp(X)
end

"""
    absolute_consumption(state, c)
"""
function absolute_consumption(state::Real, c::Real)
    return c
end

"""
    standard_budget_constraint(W, c, ω, R_e, R_base)
"""
function standard_budget_constraint(W, c, ω, R_e, R_base)
    return (1.0 - c) * W * (dot(ω, R_e) + R_base)
end

"""
    log_budget_constraint(X, c, ω, R_e, R_base)
"""
function log_budget_constraint(X, c, ω, R_e, R_base)
    port_return = dot(ω, R_e) + R_base
    if port_return <= 0.0
        return -Inf
    end
    return X + log(1.0 - c) + log(port_return)
end


"""
    make_ce_crra_extrapolator(W_min::Float64, W_max::Float64)

Creates an extrapolation strategy for when the grid stores the Certainty Equivalent (CE).
For CRRA utility, CE scales perfectly linearly with wealth.
"""
function make_ce_crra_extrapolator(W_min::Float64, W_max::Float64)
    return function(W_next::Real, Z_next, CE_next_interp)
        W_next = max(W_next, 1e-10) # Prevent 0
        if W_next < W_min
            ce_bound = CE_next_interp(W_min, Z_next...)
            return ce_bound * (W_next / W_min)
        elseif W_next > W_max
            ce_bound = CE_next_interp(W_max, Z_next...)
            return ce_bound * (W_next / W_max)
        else
            return CE_next_interp(W_next, Z_next...)
        end
    end
end

"""
    make_ce_log_crra_extrapolator(X_min::Float64, X_max::Float64)
"""
function make_ce_log_crra_extrapolator(X_min::Float64, X_max::Float64)
    return function(X_next::Real, Z_next, CE_next_interp)
        if X_next < X_min
            ce_bound = CE_next_interp(X_min, Z_next...)
            return ce_bound * exp(X_next - X_min)
        elseif X_next > X_max
            ce_bound = CE_next_interp(X_max, Z_next...)
            return ce_bound * exp(X_next - X_max)
        else
            return CE_next_interp(X_next, Z_next...)
        end
    end
end