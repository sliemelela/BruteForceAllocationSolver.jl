# Theory
In this package we are treating an investment consumption optimization problem.
More specifically, we consider consumption portfolio choice problems at timesteps
$n = 1, 2, \ldots, M$, where $M + 1$ is some terminal timestep.
This consumption portfolio choice problem at timestep $n$ is defined by an investor who maximizes the sum
of their (discounted) expected utilities over the time(step) horizon $\{n, n+1, \ldots, M+1\}$.
This is done by consumption whilst trading $N$ risky assets and a risk-free asset (cash).
Formally the investor's problem at timestep $n$ is
```math
    V_n(W_n, Z_n)
    = \max_{\{c_m, \omega_m\}_{m = n}^{M}}
        \mathbb{E}_n\left[\sum_{m = n}^{M + 1} \beta^{m - n} u(C_m)\right]

```
subject to a sequence of state transitions (budget constraints).
A standard multiplicative budget constraint takes the form:
```math
    W_{m + 1} = (1 - c_m) W_m (\omega_m^\top R^e_{m + 1} + R_{m + 1})
```
for all $m \geq n$.
Here $R^e_{m + 1}$ can be interpreted as the excess return of the risky assets over the risk-free
asset, and $R_{m + 1}$ is the gross return of other processes that *may* depend on current wealth
$(1 - c_m) W_m$ after consumption.
Furthermore, $\{c_m, \omega_m\}_{m=n}^{M}$ are the sequence of consumption controls and portfolio
weights chosen at times $m = n, \ldots, M$, and $u$ is the investor's pure utility function evaluated
on absolute physical consumption $C_m$.
It is assumed that $c_{M + 1} = 1$ (100% consumption) as it is the terminal timestep.
The process $Z_n$ is a vector of state variables that are relevant for the investor's decision making.
Lastly, $\beta \in (0,1]$ is the subjective discount factor.

### Architecture: Total Abstraction
While the equations above represent the standard formulation,
**a core feature of this package is that state variables, budget constraints, and consumption rules
are entirely abstracted.**

* **State Evolution:** The package does not enforce the standard budget constraint.
    Users inject a custom `budget_constraint` strategy
    (such as the built-in `standard_budget_constraint` or `log_budget_constraint` for models operating in log-wealth space).

* **Consumption Rules:** The control $c_m$ can represent a fraction of wealth, a fraction of log-wealth,
    or an absolute dollar amount.
    This relationship is abstracted via an injected `compute_consumption` strategy
    (e.g., `fractional_consumption` or `absolute_consumption`), which translates
    the control and state into the physical consumption $C_m$ evaluated by $u(C_m)$.

### Wealth-Dependent Returns
A key feature of this implementation is that the gross return $R_{m+1}$ is not restricted to be
exogenous. We allow $R_{m+1}$ to depend on the current level of wealth $(1 - c_m) W_m$ through the
following structure:
```math
    R_{m+1} = X_m + \frac{Y_m}{(1 - c_m)W_m}
```
where $X_m$ and $Y_m$ are functions of state variables contained in $Z_m$.
This formulation is particularly powerful as it allows the algorithm to incorporate
**non-tradeable income or fixed costs**.
For instance, if $Y_m$ represents labor income, the budget constraint correctly captures that
income is added to wealth regardless of the consumption portfolio choice
$c_m, \omega_m$.

#### Example: Including Income
Consider an investor who receives a stochastic labor income $O_n$.
Let $R^f_n$ be the gross risk-free rate.
The wealth at time $n+1$ is:
```math
    W_{n+1} = (1 - c_n) W_n (\omega_n^\top R^e_{n+1} + R^f_{n + 1}) + O_n
```
By setting $X_n = R^f_n$ and $Y_n = O_n$,
this matches our budget constraint $W_{n+1} = (1 - c_n) W_n (\omega_n^\top R^e_{n+1} + R_{n+1})$.

## Goal and Main Assumption
We assume that the returns $R^e_{m + 1}$ and $R_{m + 1}$ are driven by normally distributed shocks.
More precisely, at timestep $n$, the uncertainty realized at timestep $n+1$ is captured by a
multidimensional random shock vector $\varepsilon_{n+1}$ drawn from a multivariate
normal distribution $\mathcal{N}(0, \Sigma)$.
Because the next-period state variables $Z_{n+1}$ and the returns are fully determined by these shocks,
the investor's expectation $\mathbb{E}_n[\cdot]$ can be evaluated by integrating
directly over this multivariate normal distribution.

The goal of this package is to find the optimal $\{c_m, \omega_m\}_{m=1}^{M}$.

## How to solve continuous time problems with this package
### Setting the Scene
Suppose you are working with a continuous time model of this problem.
For example, suppose we consider an economic agent that is endowed with initial wealth
$w_0$ at time $t = 0$.
They have access to a financial market based on the non-tradeable processes: the short-term
*nominal* interest rate $r$ and the inflation rate $\pi$.

We will first consider the dynamics of stochastic processes relevant for describing the financial
market, after which we will introduce the dynamic portfolio choice problems.

We introduce the following stochastic processes:
```math
\begin{aligned}
  d r_t   &= \kappa_r(\overline{r} - r_t)\,d t + \sigma_r d Z^r_t, \\
  d \pi_t &= \kappa_\pi(\overline{\pi} - \pi_t)\,d t + \sigma_\pi d Z^\pi_t, \\
  d \Pi_t &= \Pi_t \pi_t d t, \\
  d B_t   &= r_t B_t\,d t, \\
  \frac{d S_t}{S_t}  &= \left(a r_t + b \pi_t + \lambda_S \sigma_S\right) d t
                + \sigma_S d Z^S_t, \\
  \frac{d P_t(T)}{P_t(T)} &= \left(r_t - \lambda_r \sigma_r B_r(T - t) \right) d t
    - B_r(T - t) \sigma_r d Z^r_t, \\
  \frac{d P^I_t(T)}{P^I_t(T)}
    &= \left(r_t - \lambda_r \sigma_r B_r(T - t) + \lambda_\pi \sigma_\pi B_\pi(T - t) \right) d t
        - B_r(T - t) \sigma_r d Z^r_t + B_\pi(T - t) \sigma_\pi d Z^\pi_t.
\end{aligned}
```
Writing the brownian motions as $Z = (Z^r, Z^\pi, Z^S)$,
we assume that $d \langle Z, Z \rangle_t = \rho d t$, where $\rho$ is the correlation coefficient matrix.
Furthermore we assume that the parameters
```math
    r_0, \pi_0, S_0, B_0, \overline{r}, \overline{\pi}, a, b, \lambda_r,
        \lambda_\pi, \lambda_S \in \mathbb{R},
```
and $\kappa_r, \kappa_\pi, \sigma_r, \sigma_\pi, \sigma_S \geq 0$ are exogenously given.
The functions $B_r(\cdot)$ and $B_\pi(\cdot)$ are defined as
$B_i(h) = (1 - e^{-\kappa_i h})/\kappa_i$ for $i \in \{r, \pi\}$.

As before, $r_t$ and $\pi_t$ are the short-term nominal interest rate and inflation rate,
respectively.
The process $\Pi_t$ is the consumer price index, $B_t$ is the value of the risk-free asset,
$S_t$ is the value
of the (risky) stock, $P_t(T)$ is the value of a nominal zero-coupon bond with maturity $T$, and
$P^I_t(T)$ is the value of an inflation-linked zero-coupon bond with maturity $T$.

Suppose now that the wealth dynamics $W_t$ are given by
```math
    d W_t^\omega =
        W_t^\omega \left[\omega^N_t \frac{d P_t(T)}{P_t(T)}
        + \omega^I_t \frac{d P^I_t(T)}{P^I_t(T)}
        + \omega^S_t \frac{d S_t}{S_t}
        + (1 - \omega^N_t - \omega^I_t - \omega^S_t) \frac{d B_t}{B_t}
        \right]
        + O_t d t
        - c_t d t,
```
where $\omega = (\omega^N, \omega^I, \omega^S)$ are the proportions of financial wealth invested
in the nominal bond, inflation-linked bond, and stock, respectively.
Finally $c_t$ is the consumption rate at time $t$.

Now, to add to this, suppose we are not interested in including nominal wealth, but *real* wealth
in the utility function. To that end, let us note that dynamics of real wealth
$\tilde{W}_t \coloneqq W_t/\Pi_t = (F_t + h H_t)/\Pi_t$ is given by
```math
    d \tilde{W}_t^\omega =
        \tilde{W}_t^\omega \left[\omega^N_t \frac{d P_t(T)}{P_t(T)}
        + \omega^I_t \frac{d P^I_t(T)}{P^I_t(T)}
        + \omega^S_t \frac{d S_t}{S_t}
        + (1 - \omega^N_t - \omega^I_t - \omega^S_t) \frac{d B_t}{B_t}
        - \pi_t dt
        \right]
        + \tilde{O}_t d t
        - \tilde{c}_t d t,
```
where $\tilde{O}_t = O_t/\Pi_t$ and $\tilde{c}_t = c_t/\Pi_t$.

### Transforming it into this framework
Using the Euler discretization scheme with step size $\delta t > 0$, we have that
```math
\begin{aligned}
    \tilde{W}_{n+1} &= (1 - c_n)\tilde{W}_n\bigl(\omega_n^\top R^e_{n+1} + R_{n+1}\bigr) \\
    R^e_{n+1} &=
    \begin{pmatrix}
        P_n(T)^{-1}\Delta P_{n+1}(T) - B_n^{-1}\Delta B_{n+1} \\[4pt]
        P^I_n(T)^{-1}\Delta P^I_{n+1}(T) - B_n^{-1}\Delta B_{n+1} \\
        S_n^{-1}\Delta S_{n+1} - B_n^{-1}\Delta B_{n+1}
    \end{pmatrix}, \\[6pt]
    R_{n+1} &= 1 + B_n^{-1}\Delta B_{n+1} - \pi_n\,\delta t + \frac{\tilde{O}_n}{W_n}\delta t,
\end{aligned}
```
where
```math
    \begin{aligned}
    X_{n} &=  B_n^{-1}\Delta B_{n+1} - \pi_n\,\delta t \\
    Y_n   &=  B_n^{-1}\Delta B_{n+1} - \pi_n\,\delta t
    \end{aligned}
```
where $\Delta L_{n+1} = L_{n+1} - L_t$ for any process $X_t$.
In this case, we have
```math
\begin{aligned}
    B_n^{-1}\Delta B_{n+1}
    &= r_n \delta t, \\
    P_n(T)^{-1}\Delta P_{n+1}(T)
    &= \left(r_n - \lambda_r \sigma_r B_r(T - n\delta t) \right) \delta t
        - B_r(T - n\delta t) \sigma_r \varepsilon^r_{n+1}, \\
    P^I_n(T)^{-1}\Delta P^I_{n+1}(T)
    &= \left(r_n - \lambda_r \sigma_r B_r(T - n\delta t) +
            \lambda_\pi \sigma_\pi B_\pi(T - n\delta t) \right) \delta t \\
        &\phantom{=}
        - B_r(T - n\delta t) \sigma_r \varepsilon^r_{n+1}
        + B_\pi(T - n\delta t) \sigma_\pi \varepsilon^\pi_{n+1}, \\
    S_n^{-1}\Delta S_{n+1}
    &= \left( a r_n + b \pi_n + \lambda_S \sigma_S\right) \delta t
        + \sigma_S \varepsilon^S_{n+1},
\end{aligned}
```
where $\varepsilon_{n+1} = (\varepsilon^r_{n+1}, \varepsilon^\pi_{n+1}, \varepsilon^S_{n+1})$ are
identically distributed over time, drawn from a multivariate normal distribution with mean zero
and covariance matrix $\rho \delta t$.
(Note: Users can package these transition dynamics cleanly into closures, such as the provided
 `make_merton_transition` function).

## Algorithm
We use backwards recursion to solve for the optimal $\{c_m, \omega_m\}_{m=1}^{M}$.
This means that at some timestep $n$, we assume that $V_{n + 1}(W_{n + 1}, Z_{n + 1})$ is known.
The base case is that $V_{M + 1}(W_{M + 1}, Z_{M + 1}) = u(W_{M + 1})$.

To fully understand the algorithm we must first treat two propositions that are used in it:
The Bellman Equation and Gauss-Hermite Quadrature.

### The Bellman Equation
We first derive the generalized form of the Bellman equation for the value function.

The value function satisfies the Bellman equation
```math
    V_m(W_n, Z_n) = \max_{c_n, \omega_n}
        \left\{
            u(C_n)
            + \beta \mathbb{E}_n\left[
                V_{n + 1}(W_{n + 1}, Z_{n + 1})
            \right]
        \right\},
```

with terminal condition $V_{M + 1}(W_{M + 1}, Z_{M + 1}) = u(C_{M + 1})$ and
state transitions defined by the chosen `budget_constraint`.

#### Proof
We have that

```math
\begin{aligned}
    V_n(W_n, Z_n)
    &= \max_{c_n, \omega_n} \max_{\{c_m, \omega_m\}_{m = n + 1}^{M}} \left\{
        u(C_n)
        + \beta\mathbb{E}_n\left[
            \sum_{m = n + 1}^{M + 1} \beta^{m - (n + 1)} u(C_m)
        \right]
    \right\} \\
    &= \max_{c_n, \omega_n} \left\{
        u(C_n)
        + \beta \mathbb{E}_n\left[
            \max_{\{c_s, \omega_s\}_{s = t + 1}^{M}}
                    \sum_{m = n + 1}^{M + 1} \beta^{m - (n + 1)} u(C_m)
        \right]
    \right\} \\
    &= \max_{c_n, \omega_n}
        \left\{
            u(C_n)
            + \beta \mathbb{E}_n\left[
               \max_{\{c_m, \omega_m\}_{m = n + 1}^{M}}
                    \mathbb{E}_{n + 1}\left[\sum_{m = n + 1}^{M + 1} \beta^{m - (n + 1)} u(C_m)\right]
            \right]
        \right\} \\
    &= \max_{c_n, \omega_n}
        \left\{
            u(C_n)
            + \beta \mathbb{E}_n\left[
                V_{n + 1}(W_{n + 1}, Z_{n + 1})
            \right]
        \right\}.
\end{aligned}

```

The first equality follows from the definition of the value function.
The second equality follows from the fact that the maximization over
$\{c_m, \omega_m\}_{m = n + 1}^{M}$
is "independent" of the choices at time $n$ and can thus be moved inside the expectation.
The third equality follows from the tower property of conditional expectations.
The last equality follows from the definition of the value function at time $n + 1$.

### Gauss-Hermite Quadrature

Using the Bellman equation, we are now tasked to be evaluate

```math
    \mathbb{E}_n\left[V_{n + 1}(W_{n + 1}, Z_{n + 1})\right]
    = \underbrace{\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty}}_{D \text{ times}}
        V_{n+1}\Big(W_{n+1}(\boldsymbol{\varepsilon}), Z_{n+1}(\boldsymbol{\varepsilon})\Big)
        f(\boldsymbol{\varepsilon}) \, d\varepsilon_1 \cdots d\varepsilon_D,

```

where $f(\boldsymbol{\varepsilon})$ is the joint probability density function of the $D$-dimensional
shock vector $\boldsymbol{\varepsilon}$.
Since we assume that the randomness in the model is driven by purely normally distributed shocks
we must compute an integral over the
multivariate normal distribution of the shocks.
Instead of relying on computationally expensive Monte Carlo simulations, we approximate this
integral using Gauss-Hermite quadrature.

The standard Gauss-Hermite quadrature rule approximates integrals of the form
$\int_{-\infty}^{\infty} f(x) e^{-x^2} dx$ using a set of $Q$ deterministic nodes $x_k$ and
corresponding weights $\omega_k$ (not to be confused with the portfolio choice weights)

```math
\int_{-\infty}^{\infty} f(x) e^{-x^2} d x \approx \sum_{k=1}^{Q} \omega_k f(x_k).

```

How these $x_k$ and $\omega_k$ are chosen is outside of the scope of this package and we
simply employ a package that computes these for us.
For the curious reader: the nodes and weights are deterministically derived from the roots of
the so-called Hermite polynomials and appropriately scaled to guarantee that the expected value of
any polynomial up to degree $2Q-1$ under a standard normal distribution is calculated with perfect
mathematical exactness.

We do run into the issue, however, that the expectation $\mathbb{E}_n[\cdot]$ is taken with
respect to a standard normal probability density function $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$.
To map the standard normal integral into the Gauss-Hermite framework, we apply the change of
variables $z = x\sqrt{2}$. This yields the transformed standard normal nodes $z_k$ and weights $w_k$:

```math
    z_k = x_k \sqrt{2}, \quad w_k = \frac{\omega_k}{\sqrt{\pi}},

```

such that $\int_{-\infty}^{\infty} g(z) \phi(z) dz \approx \sum_{k=1}^{Q} w_k g(z_k)$.

#### Multivariate Extension and Correlation

Because our model relies on a multidimensional shock vector, we extend this to $D$ dimensions by
taking the Cartesian product of the 1D nodes and weights.
This results in $Q^D$ independent multidimensional nodes
$\boldsymbol{z}_j = (z_{j_1}, \ldots, z_{j_D})^\top$ and their corresponding joint weights
$W_j = \prod_{d=1}^D w_{j_d}$.
As stated in our main assumptions, the random shock vector $\varepsilon_{n+1}$ is drawn from
$\mathcal{N}(0, \Sigma)$.
To introduce the correct covariance structure to our independent nodes $\boldsymbol{z}_j$,
we utilize the Cholesky decomposition of the covariance matrix
$\Sigma = L L^\top$, where $L$ is a lower triangular matrix.
The correlated multidimensional shocks at node $j$ are given by

```math
    \boldsymbol{\varepsilon}_j = L \boldsymbol{z}_j.

```

By the Law of the Unconscious Statistician, we can evaluate the expectation of the
future value function by integrating over the distribution of the shocks
rather than the distributions of the future states.
Thus, the expectation in the Bellman equation is approximated by the discrete sum:

```math
    \mathbb{E}_n\left[ V_{n + 1}(W_{n + 1}, Z_{n + 1}) \right]
    \approx \sum_{j=1}^{Q^D} W_j \cdot V_{n+1}\Big(W_{n+1}(\boldsymbol{\varepsilon}_j), Z_{n+1}(\boldsymbol{\varepsilon}_j)\Big),

```

where $W_{n+1}(\boldsymbol{\varepsilon}_j)$ and $Z_{n+1}(\boldsymbol{\varepsilon}_j)$
are the realized next-period wealth and state variables given the shock
$\boldsymbol{\varepsilon}_j$.
This interpolated value $\hat{V}_{n+1}$ is then used directly in the Bellman maximization step.

### Grid Selection

Before the algorithm runs, we first do a forward simulation to be able to choose appropriate
grids for the state space $Z_m = (Z^1_m, \ldots, Z^K_m)$ and principal state space $W_m$ at each time $m = 1, \ldots, M$.
To obtain values for $W_m$, we use the simple strategy $c_m = 0$ and $\omega_m = 0$
for all $m = 1, \ldots, M$. We then choose a *fixed* size for the grids.

#### State variables
Let $G_z, G_w \in \mathbb{N}$ denote the sizes of the state space grid and principal state
(wealth) space grid respectively.
After the forward simulation provides the range of possible outcomes, we define the discretization.

For each auxiliary state variable $Z^{i}_n$, we construct a grid between $Z^{i}_{n, \min}$ and $Z^{i}_{n, \max}$. While a uniform linear grid is standard, the package provides several built-in generators for the principal state variable $W_n$:

* `generate_linear_grid`: Generates a standard linear grid.
* `generate_log_spaced_grid`: Generates a grid spaced uniformly in logs but evaluated in absolute levels, effectively clustering points near zero to capture high curvature.
* `generate_adaptive_grid`: Automatically calculates the curvature (second derivative) of the utility function and distributes grid points proportional to $\sqrt{|u''(W)|}$.

#### Returns

For a given state $Z_n$ and shock $\boldsymbol{\varepsilon}_j$,
the realized returns are calculated using the model-specific transition functions

```math
R^e_{n+1, j} = \mathcal{R}^e(Z_n, \boldsymbol{\varepsilon}_j), R_{n+1, j} = \mathcal{R}(Z_n, W_n, c_n, \boldsymbol{\varepsilon}_j)

```

Commonly, for Geometric Brownian Motion, this takes the form

```math
R_{n+1, j} = \exp\left( (\mu(Z_n) - \frac{1}{2}\sigma^2) \Delta t + \sigma \sqrt{\Delta t} \boldsymbol{\varepsilon}_j \right).

```

#### Portfolio Weights and Consumption Fractions

Similarly, let $G_\omega, G_c \in \mathbb{N}$ denote the sizes of the portfolio weights and
consumption control grid respectively, and write $\omega = (\omega^1, \ldots, \omega^N)$.
The standard values we take for boundaries are $\omega_{\min} = -1, \omega_{\max}=2, c_{\min} = 0, c_{\max} = 1$.
Then

```math
    \begin{aligned}
    \mathcal{G}_{\omega^{i}, n}
        &= \left\{ \omega^{i}_{\min} + (j - 1) \cdot \frac{\omega^{i}_{\max} - \omega^{i}_{\min}}{G_\omega - 1} \right\}_{j=1}^{G_\omega}, \\
    \mathcal{G}_{c, n}
        &= \left\{ c_{\min} + (j - 1) \cdot \frac{c_{\max} - c_{\min}}{G_c - 1} \right\}_{j=1}^{G_c},
    \end{aligned},

```

for all $n = 1, \ldots, M$.

### Backwards Recursion

We proceed iteratively backwards from $n = M$ down to $1$.
Assuming the next-period value function $V_{n+1}$ is known
(with the terminal condition $V_{M+1}(W_{M+1}, Z_{M+1}) = u(C_{M+1})$),
we evaluate the optimal policy for every state combination
$(W_n, Z_n) \in \mathcal{G}_{W, n} \times \mathcal{G}_{Z, n}$ through the following steps:

#### Control Loop

For the current state $(W_n, Z_n)$, iterate over all candidate consumption controls
$c_n \in \mathcal{G}_{c, n}$ and portfolio weights $\omega_n \in \mathcal{G}_{\omega, n}$.

#### Quadrature Integration

For each control pair $(c_n, \omega_n)$, approximate the expected future value using the
$Q^D$ multidimensional quadrature nodes $\boldsymbol{z}_j$ and weights $W_j$:

* Generate the correlated shock vector $\boldsymbol{\varepsilon}_j = L \boldsymbol{z}_j$.
* Evaluate the realized returns $R^e_{n+1}(\boldsymbol{\varepsilon}_j)$ and
$R_{n+1}(\boldsymbol{\varepsilon}_j)$ using the model's transition dynamics.
* Compute the realized next-period wealth and state variables:

```math
    \begin{aligned}
    W_{n + 1, j} &= \text{budget\_constraint}(W_n, c_n, \omega_n, R^e_{n+1}, R_{n+1}) \\
    Z_{n + 1, j} &= f_Z(Z_n, \boldsymbol{\varepsilon}_j),
    \end{aligned}

```

where $f_Z$ denotes the deterministic state transition equations defined by the discretized SDEs.

#### Bellman Maximization and Interpolation

Calculate the current objective value for these controls and select the maximum

```math
    V_n(W_n, Z_n) \approx \max_{c_n, \omega_n} \left\{ u(C_n) + \beta \sum_{j=1}^{Q^D} W_j \cdot \hat{V}_{n+1}(W_{n+1, j}, Z_{n+1, j}) \right\}

```
#### Interpolation and Extrapolation
If the maximum has been calculated for all grid points $\mathcal{G}_{W, n} \times \mathcal{G}_{Z, n}$,
use multidimensional linear interpolation to create a continuous function $\hat{V}_{n}(W_{n}, Z_{n})$.

Crucially, what happens if a simulated market shock causes $W_{n+1, j}$ to evaluate to a point
*outside* the defined boundaries of the grid $\mathcal{G}_{W, n}$?

Standard linear extrapolation fails catastrophically for highly curved utility functions like CRRA.
By simply continuing the slope from the edge of the grid, linear extrapolation creates two silent economic failures:
1. **The Upside Trap:** As wealth goes to infinity, marginal utility should decay to zero.
    A straight line, however, shoots upward indefinitely, potentially crossing into positive utility
    (which is mathematically invalid for $\gamma > 1$). The agent perceives infinite reward for taking infinite risk.
2. **The Downside Trap:** As wealth approaches zero, the penalty should plummet exponentially
    toward $-\infty$. A straight line gently slopes downward,
    causing the agent to massively underestimate the penalty of bankruptcy and take reckless leverage.

To solve this, the package abstracts extrapolation via user-provided strategies that abandon linear
rays and instead mathematically "stitch" the exact asymptotic economic curvature onto the grid boundaries.
We provide two standard closures for this:
* **`make_crra_extrapolator` (Absolute Wealth):** Leverages the scale-invariance property of CRRA utility. If the agent's wealth falls outside the grid at $W_{bound}$, the strategy evaluates the known value at the boundary $V(W_{bound})$ and scales it exactly by the CRRA ratio:
```math
V(W_\text{next}) = V(W_\text{bound}) \times \left( \frac{W_\text{next}}{W_\text{bound}} \right)^{1-\gamma}
```

* **`make_log_crra_extrapolator` (Log-Wealth):** When the state variable is formulated in
    logs ($X = \log W$), the value function takes the shape of a steep exponential curve
    $V(X) \propto e^{(1-\gamma)X}$. Linear extrapolation in log-space is equally dangerous.
    This strategy translates the CRRA boundary scaling factor into log-space, applying an
    exponential penalty or decay:
```math
V(X_\text{next}) = V(X_\text{bound}) \times \exp\Big((1-\gamma)(X_\text{next} - X_\text{bound})\Big)
```

By wrapping these mathematical rules in closures and injecting them into the Bellman objective,
the algorithm ensures perfect numerical stability and rational economic behavior even under extreme simulated market shocks.

#### Policy Storage

Record the interpolated value function and the $c_n^*$ and $\omega_n^*$
that maximize the Bellman equation as the optimal policy for the current grid point $(W_n, Z_n)$.

## Notes

On brute force:

* Can we not do the same procedure but then use some hill-climbing algorithm?
More precisely, For the current fixed state $(W_n, Z_n)$, we define a continuous
objective function $f(c_n, \omega_n)$ that represents the right-hand side of the Bellman equation.
That would perhaps scale better with higher dimensions as well.
* I can't exactly recall the "zooming" procedure that Servaas mentioned earlier.