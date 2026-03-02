# Testing and Validation
Numerical dynamic programming algorithms for portfolio choice are highly sensitive to implementation details.
Minor errors in quadrature integration, grid interpolation, or
Bellman maximization often do not result in runtime crashes, but rather in silently incorrect economic behavior.

To ensure the integrity of this package, we validate the numerical outputs against four
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

## Test Case 3: Complete Market without Human Capital (Real Wealth Optimization)
This test validates the algorithm's ability to handle multidimensional continuous-time processes and
inflation-linked assets in a complete market setting where exact analytical solutions are known.
This serves as the direct baseline before introducing non-tradeable income constraints.

**Setup:**

* **Assets:** A complete financial market consisting of a risk-free bank account, a risky stock, a nominal zero-coupon bond, and an inflation-linked zero-coupon bond.
* **State Variables ($Z$):** The short-term nominal interest rate $r_t$ and the inflation rate $\pi_t$. Both follow correlated mean-reverting Ornstein-Uhlenbeck dynamics.
* **Wealth-Dependent Returns:** None. We set $H_t = 0$.
* **Utility Function:** CRRA utility derived strictly from *real* terminal wealth at time $T$, evaluated as $u(W_T/\Pi_T) = \frac{(W_T/\Pi_T)^{1-\gamma}}{1-\gamma}$.

**Expected Output:**
Because this complete market problem has an exact analytical solution,
the numerical optimal portfolio weights $\omega_t^*$ produced by the package must converge precisely to the following closed-form expressions:

1. **Stock Allocation:** The optimal weight in the risky stock must be constant and equal to:
```math
\tilde{\omega}_{t}^{S}=-\frac{\phi_{S}}{\sigma_{S}}
```
where $\phi_S$ is the market price of risk factor loading for the stock,
and $\sigma_S$ is its volatility.

2. **Inflation-Linked Bond Allocation:** The optimal weight in the inflation-linked bond must match the formula:
```math
\tilde{\omega}_{t}^{I}=1-\frac{1}{\gamma}-\frac{\phi_{\pi}}{B_{\pi}(T-t)\sigma_{\pi}}
```
where $\phi_\pi$ is the factor loading for inflation, $\sigma_\pi$ is inflation volatility, and
$B_\pi(T-t)$ is the bond's sensitivity to the inflation rate at time to maturity $T-t$.

3. **Nominal Bond Allocation:** The optimal weight in the nominal zero-coupon bond must match:
```math
\tilde{\omega}_{t}^{N}=\frac{B_{r}(T-t)\sigma_{r}\phi_{\pi}+B_{\pi}(T-t)\sigma_{\pi}\phi_{r}}{B_{r}(T-t)B_{\pi}(T-t)\sigma_{r}\sigma_{\pi}}
```
where the respective $\phi$, $\sigma$, and $B$ terms correspond to the interest rate $r$ and inflation rate $\pi$ parameters.


*Validation Check:* The outputs for all three weights should exactly match these analytical values across
the entire wealth grid $\mathcal{G}_{W, n}$ since human capital is zero and CRRA utility scales perfectly with financial wealth.

## Test Case 5: Complete Market with Stochastic Human Capital (Real Wealth Optimization)
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