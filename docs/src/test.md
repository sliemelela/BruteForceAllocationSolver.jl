# Testing and Validation
Numerical dynamic programming algorithms for portfolio choice are highly sensitive to implementation details.
Minor errors in quadrature integration, grid interpolation, or
Bellman maximization often do not result in runtime crashes, but rather in silently incorrect economic behavior.

To ensure the integrity of this package, we validate the numerical outputs against three
benchmark models with known analytical or semi-analytical properties.
Users modifying the core algorithm should run these tests to verify that their changes have not
broken fundamental economic mechanics.

## Test Case 1: The Samuelson-Merton Model (Core Mechanics)
This test strips the model down to its bare essentials to verify the Bellman recursion,
 multidimensional quadrature, and basic wealth grid interpolation.

**Setup:**
* **Assets:** One risk-free asset with a constant gross return $R^f = \exp(r \delta t)$, and one risky asset with constant expected return $\mu$ and volatility $\sigma$.
* **State Variables ($Z$):** None. The investment opportunity set is constant, meaning returns are identically and independently distributed (i.i.d.) over time.
* **Wealth-Dependent Returns:** Exogenous (i.e., $X_n = R^f$, $Y_n = 0$).
* **Utility Function:** CRRA utility $u(W) = \frac{W^{1-\gamma}}{1-\gamma}$ with risk aversion $\gamma > 1$.

**Expected Output:**
1. **Wealth Independence:** The optimal portfolio weight $\omega_n^*$ must be completely constant
    across the entire wealth grid $\mathcal{G}_{W, n}$.
2. **Convergence to the Merton Share:** In the continuous-time limit
    (as $\delta t \to 0$), the optimal portfolio weight must converge to the analytical Merton fraction:
    $$
    \omega^* \approx \frac{\mu - r}{\gamma \sigma^2}
    $$
3. **Consumption:** The optimal consumption fraction $c_n^*$ must also be independent of the
    wealth level, though it will monotonically increase toward $1.0$ as $n \to M$ (the terminal horizon).

## Test Case 2: Deterministic Labor Income (Wealth-Dependent Returns)
This test validates the budget constraint logic and the `Extension: Wealth-Dependent Returns`
feature by introducing a non-tradeable cash flow.

**Setup:**
* **Assets and Utility:** Identical to Test Case 1 (constant $r, \mu, \sigma$, and CRRA utility).
* **Wealth-Dependent Returns:** We introduce a constant, guaranteed future income stream.
    We set $X_n = R^f$ and $Y_n = Y$ for some constant $Y > 0$.
**Expected Output:**
1. **Human Capital as a Bond:** Because the investor has guaranteed future income (
    acting as an implicit, risk-free bond holding), they should take *more* risk with their
    liquid financial wealth compared to Test Case 1.
2. **Wealth-Dependent Portfolio Weights:** The optimal portfolio weight $\omega_n^*$ must
    now be strictly decreasing in $W_n$:
* **Low Wealth:** At grid points near $W = 0$, the future income dominates the investor's total net worth. $\omega_n^*$ should be highly leveraged, likely hitting the upper bound constraint $\omega_{\max}$.
* **High Wealth:** As $W_n \to \infty$, the labor income becomes mathematically negligible relative to financial wealth. The optimal weight $\omega_n^*$ must asymptotically approach the standard Merton share $\frac{\mu - r}{\gamma \sigma^2}$.



## Test Case 3: Predictable Returns and Hedging (State Variables)

This test validates the multidimensional state grid $\mathcal{G}_{Z, n}$, the multidimensional interpolation, and the Cholesky correlation matrix step inside the Gauss-Hermite quadrature.

**Setup:**

* **State Variable ($Z_n$):** A mean-reverting AR(1) or Ornstein-Uhlenbeck process representing a return predictor (e.g., a dividend yield).
* **Asset Returns:** The risk-free rate is constant. The risky asset's expected return is time-varying and equal to the state variable: $\mu_n = Z_n$.
* **Correlation:** The shocks to the state variable $Z_n$ and the risky asset returns are negatively correlated ($\rho < 0$).

**Expected Output:**

1. **State-Dependent Weights (Market Timing):** The optimal weight $\omega_n^*$ must positively correlate with the state grid $\mathcal{G}_{Z, n}$. Higher values of $Z_n$ imply higher expected returns, resulting in larger allocations to the risky asset.
2. **Intertemporal Hedging Demand:** Because of the negative correlation between the asset returns and the state variable (investment opportunities), a long-term investor will exhibit hedging demands. The optimal $\omega_n^*$ produced by the algorithm must be *higher* than the allocation chosen by a myopic (single-period) investor.
3. **Correlation Sensitivity:** If the user artificially sets the correlation parameter $\rho = 0$, the intertemporal hedging demand should disappear entirely.


## Test Case 4: Complete Market with Stochastic Human Capital (Real Wealth Optimization)
This advanced test validates the algorithm's ability to handle multidimensional
continuous-time processes, inflation-linked assets, and a highly structured stochastic income stream
acting as human capital.

**Setup:**

* **Assets:** A complete financial market consisting of a risk-free bank account, a risky stock, a nominal zero-coupon bond, and an inflation-linked zero-coupon bond.


* **State Variables ($Z$):** The short-term nominal interest rate $r_t$ and the inflation rate $\pi_t$. Both follow correlated mean-reverting Ornstein-Uhlenbeck dynamics.


* **Wealth-Dependent Returns (Income):** The agent receives a stochastic outside income stream (e.g., labor income) that grows with the inflation rate (commodity price level $\Pi_t$). The discounted value of this future income represents the agent's human capital $H_t$.


* **Utility Function:** CRRA utility derived strictly from *real* terminal wealth at time $T$, evaluated as $u(W_T/\Pi_T) = \frac{(W_T/\Pi_T)^{1-\gamma}}{1-\gamma}$.



**Expected Output:**

1. **Increased Stock Market Exposure:** Because human capital represents a large, non-tradeable asset that is not exposed to stock market risk, the optimal policy must allocate a larger share of the remaining *financial* wealth into the risky stock to achieve the desired overall risk exposure.

2. **Life-Cycle Glidepath:** As the individual ages towards the terminal horizon $T$, their remaining human capital decreases relative to their financial wealth. Consequently, the share of financial wealth invested in the risky stock must steadily decrease over time.

3. **Inflation Hedging Substitution:** Because the stochastic labor income already acts as an intrinsic hedge against inflation, the optimal portfolio weight for the inflation-linked bond must be strictly *lower* than it would be in a baseline model without human capital.

4. **Interest Rate Hedging:** To offset the risks and substitute the inflation bond, the optimal policy must allocate *more* financial wealth into the nominal zero-coupon bond to hedge against short-term interest rate risk.