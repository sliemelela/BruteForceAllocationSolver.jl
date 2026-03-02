# Theory
In this package we are treating an investment consumption optimization problem.
More specifically, in this package we consider consumption portfolio choice problems at timesteps
$n = 1, 2, \ldots, M$, where $M + 1$ is some terminal timestep.
This consumption portfolio choice problem at timestep $n$ is defined by an investor who maximizes the sum
of their (discounted) expected utilities over the time(step) horizon $\{n, n+1 \ldots, M+1\}$.
This is done by trading $N$ risky assets and a risk-free asset (cash).
Formally the investor's problem at timestep $n$ is
```math
    V_n(W_n, Z_n)
    = \max_{\{c_m, \omega_m\}_{m = n}^{M}}
        \mathbb{E}_n\left[\sum_{m = n}^{M + 1} \beta^{m - n} u(c_m W_m)\right]
```

subject to the sequence of budget constraints
```math
    W_{m + 1} = (1 - c_m) W_m (\omega_m^\top R^e_{m + 1} + R_{m + 1})
```
for all $m \geq n$.
Here $R^e_{m + 1}$ can be interpreted as the excess return of the risky assets over the risk-free
asset, and $R_{m + 1}$ is the gross return of other processes that _may_ depend on current wealth
$(1 - c_m) W_m$.
Furthermore, $\{c_m, \omega_m\}_{m=n}^{M}$ are the sequence of consumption _fractions_
(i.e. $c_m \in [0,1]$ for all $m$) and
portfolio weights chosen at times $m = n, \ldots, M$ and $u$ is the investor's utility function.
It is assumed that $c_{M + 1} = 1$ as it is the terminal timestep.
The process $Z_n$ is a vector of state variables that are relevant for the investor's decision making.
Lastly, $\beta \in (0,1]$ is the subjective discount factor.
The goal of this package is to find the optimal $\{c_m, \omega_m\}_{m=1}^{M}$.

### Extension: Wealth-Dependent Returns
A key feature of this implementation is that the gross return $R_{m+1}$ is not restricted to be
exogenous.
We allow $R_{m+1}$ to depend on the current level of wealth $(1 - c_m) W_m$ through the
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

#### Example: Solving a Continuous Time Problem optimizing for Real Wealth

##### Setting the Scene
Suppose you are working with a continuous time model of this problem.
For example, suppose we consider an economic agent that is endowed with initial wealth \
$w_0$ at time $t = 0$.
They have access to a financial market based on the non-tradeable processes: the short-term
_nominal_ interest rate $r$ and the inflation rate $\pi$.

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
        \lambda_\pi, \lambda_S \in \R,
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

Now, to add to this, suppose we are not interested in including nominal wealth, but _real_ wealth
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

##### Transforming it into this framework
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
i.i.d. normal random variables with mean zero and covariance matrix $\rho \delta t$.

## Algorithm
We use backwards recursion to solve for the optimal $\{c_m, \omega_m\}_{m=1}^{M}$.
This means that at some timestep $n$, we assume that $V_{n + 1}(W_{n + 1}, Z_{n + 1})$ is known.
The base case is that $V_{M + 1}(W_{M + 1}, Z_{M + 1}) = u(W_T)$.


### The Bellman Equation
We first derive the generalized form of the Bellman equation for the value function.

The value function satisfies the Bellman
equation
```math
    V_m(W_n, Z_n) = \max_{c_n, \omega_n}
        \left\{
            u(c_n W_n)
            + \beta \mathbb{E}_n\left[
                V_{n + 1}(W_{n + 1}, Z_{n + 1})
            \right]
        \right\},
```
with terminal condition $V_{M + 1}(W_{M + 1}, Z_{M + 1}) = u(W_{M + 1})$ and
budget constraint as before.


#### Proof
We have that
```math
\begin{aligned}
    V_n(W_n, Z_n)
    &= \max_{c_n, \omega_n} \max_{\{c_m, \omega_m\}_{m = n + 1}^{M}} \left\{
        u(c_n W_n)
        + \beta\mathbb{E}_n\left[
            \sum_{m = n + 1}^{M + 1} \beta^{m - (n + 1)} u(c_m W_m)
        \right]
    \right\} \\
    &= \max_{c_n, \omega_n} \left\{
        u(c_n W_n)
        + \beta \mathbb{E}_n\left[
            \max_{\{c_s, \omega_s\}_{s = t + 1}^{M}}
                    \sum_{m = n + 1}^{M + 1} \beta^{m - (n + 1)} u(c_m W_m)
        \right]
    \right\} \\
    &= \max_{c_n, \omega_n}
        \left\{
            u(c_n W_n)
            + \beta \mathbb{E}_n\left[
               \max_{\{c_m, \omega_m\}_{m = n + 1}^{M}}
                    \mathbb{E}_{n + 1}\left[\sum_{m = n + 1}^{M + 1} \beta^{m - (n + 1)} u(c_m W_m)\right]
            \right]
        \right\} \\
    &= \max_{c_n, \omega_n}
        \left\{
            u(c_n W_n)
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


### Grid Selection
Before the algorithm runs, we first do a forward simulation to be able to choose appropriate
grids for the state space $Z_m = (Z^1_m, \ldots, Z^K_m)$ and wealth space $W_m$ at each time $m = 1, \ldots, M$.
The obtain values for $W_m$, we use the simple strategy $c_m = 0$ and $\omega_m = 0$
for all $m = 1, \ldots, M$.
We then choose a *fixed* size for the grids (we will set up).


#### State variables
Let $G_z, G_w \in \mathbb{N}$ denote the sizes of the state space grid and wealth space grid respectively.
After the forward simulation provides the range of possible outcomes, we define the discretization.
For each state variable $Z^{i}_n$ and the wealth variable $W_n$,
we construct a uniform linear grid.
Let $Z^{i}_{n, \min}$ and $Z^{i}_{n, \max}$ be the boundary values observed during simulation.
The grid for each variable is defined as:
```math
    \begin{aligned}
    \mathcal{G}_{Z^{i}, n}
        &= \left\{ Z^{i}_{n, \min} + (j - 1) \cdot \frac{Z^{i}_{n, \max} - Z^{i}_{n, \min}}{G_z - 1} \right\}_{j=1}^{G_z}, \\
    \mathcal{G}_{W, n}
        &= \left\{ W_{n, \min} + (j - 1) \cdot \frac{W_{n, \max} - W_{n, \min}}{G_w - 1} \right\}_{j=1}^{G_w}.
    \end{aligned}
```
for all $n = 1, \ldots, M$.

#### Returns
To evaluate the expectation $\mathbb{E}_n[V_{n+1}]$,
we do not discretize the returns themselves into a fixed grid.
Instead, we discretize the innovations (shocks) that drive the returns and state variables.
We assume that the transitions of $Z_{n+1}$ and the returns $R^e_{n+1}$ are driven by a vector of
shocks $\epsilon_{n+1} \sim \mathcal{N}(0, \Sigma)$.
We use Gauss-Hermite Quadrature to represent these shocks.

**Independent Shocks**: Let $Q$ be the number of quadrature nodes per dimension.
We obtain a set of one-dimensional nodes $\{x_q\}_{q=1}^Q$ and weights $\{w_q\}_{q=1}^Q$.
For a $D$-dimensional shock vector (where $D$ is the number of Brownian motions),
we form the Cartesian product of these nodes to get a total of $Q^D$ multidimensional nodes
$\mathbf{x}_j$ and corresponding weights $W_j$.

**Correlation Mapping**: If the shocks are correlated with covariance matrix $\Sigma$,
we apply the Cholesky decomposition $L$ (where $LL^\top = \Sigma$) to the independent nodes:
```math
\boldsymbol{\epsilon}_j = L \mathbf{x}_j
```
**Realized Returns**: For a given state $Z_n$ and shock $\boldsymbol{\epsilon}_j$,
the realized returns are calculated using the model-specific transition functions:
```math
R^e_{n+1, j} = \mathcal{R}^e(Z_n, \boldsymbol{\epsilon}_j), R_{n+1, j} = \mathcal{R}(Z_n, W_n, c_n, \boldsymbol{\epsilon}_j)
```
Commonly, for Geometric Brownian Motion, this takes the form:
```math
R_{n+1, j} = \exp\left( (\mu(Z_n) - \frac{1}{2}\sigma^2) \Delta t + \sigma \sqrt{\Delta t} \boldsymbol{\epsilon}_j \right)
```


#### Portfolio Weights and Consumption Fractions
Similarly, let $G_\omega, G_c \in \mathbb{N}$ denote the sizes of the portfolio weights and
consumption fraction grid respectively, and write $\omega = (\omega^1, \ldots, \omega^N)$.
The grid selection for portfolio weights is ad-hoc.
Choose some $\omega_{\min}, \omega_{\max} \in \mathbb{R}$.
The standard values we take are $\omega_{\min} = -1, \omega_{\max}=2$.
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
We follow the following steps at timestep $n$:
-  For each $Z_n^{i} \in \mathcal{G}_{Z^{i}, n}$ for $i = 1, \ldots, K$,
    $W_n \in \mathcal{G}_{W, n}$, $\omega_n \in \mathcal{G}_{\omega, n}$, and $c_n \in \mathcal{G}_{c, n}$
     we compute
```math
W_{n + 1} = (1 - c_n) W_n (\omega_)
```

## Notes

On the grid creation:
- Now a linear uniform grid is used, perhaps another type of spacing?
- Now we do the forward simulation using $c_m, \omega_m = 0$ for all $m$. Is there a better choice?
- Now we take the minimum and maximum of the simulated paths.
    Perhaps take 95th and 5th percentile (or something else)?