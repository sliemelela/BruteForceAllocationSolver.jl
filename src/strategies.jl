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

function make_shadow_wealth_estimator(M, dt, κ_r, overline_r, κ_π, overline_π; is_complete_market=true)
    return function(n_next, r_next, pi_next)
        time_remaining = (M - n_next + 1) * dt

        if time_remaining <= 0.0
            return 0.0
        end

        if is_complete_market
            # For Problem 1: Use your exact analytical pricing (A_I, B_r, B_π formulas)
            # You already have this logic in `get_HC_and_durations` in your scripts!
            H_exact = 0.0
            for step in n_next:M
                h = (step - n_next) * dt
                # Calculate exact real zero-coupon bond price P_real
                P_real = exp(A_I(h) - B_r(h)*r_next + B_π(h)*pi_next)
                H_exact += 1.0 * dt * P_real
            end
            return H_exact
        else
            # For Problem 3: We just need a topological shift.
            # Discounting future income at the long-term expected real rate is a great heuristic.
            expected_real_rate = overline_r - overline_π
            if abs(expected_real_rate) < 1e-6
                return 1.0 * time_remaining
            else
                return 1.0 * (1.0 - exp(-expected_real_rate * time_remaining)) / expected_real_rate
            end
        end
    end
end


function make_ce_financial_wealth_extrapolator(F_min::Float64, F_max::Float64, get_H_func::Function)
    # Notice we now accept n_next to know how much time is left!
    return function(F_next::Real, Z_next, CE_next_interp, n_next::Int)

        # 1. Get the shadow wealth to shift the singularity
        H_shadow = get_H_func(n_next, Z_next...)

        # 2. Scale using Total Shadow Wealth
        if F_next < F_min
            ce_bound = CE_next_interp(F_min, Z_next...)
            return ce_bound * ((F_next + H_shadow) / (F_min + H_shadow))

        elseif F_next > F_max
            ce_bound = CE_next_interp(F_max, Z_next...)
            return ce_bound * ((F_next + H_shadow) / (F_max + H_shadow))

        else
            return CE_next_interp(F_next, Z_next...)
        end
    end
end

@inline function _invoke_extrapolator(extrapolator, W_next, Z_next, CE_next, n_next)
    # Check if the extrapolator accepts the 4th argument (n_next)
    if applicable(extrapolator, W_next, Z_next, CE_next, n_next)
        return extrapolator(W_next, Z_next, CE_next, n_next)
    else
        # Fallback to the legacy 3-argument signature
        return extrapolator(W_next, Z_next, CE_next)
    end
end