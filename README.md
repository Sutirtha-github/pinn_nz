# **Deep Learning the Nakajima-Zwanzig Equation Of Motion using Physics Informed Neural Network**

<br><br>

## Theory

<br>

### The Volterra equation:

The **Volterra-integro-differential equation** is a special type of equation in which the unknown function appears as a combination of ordinary derivatives and under the integral sign attached to an arbitrary kernel function. The generalized equation takes on the form,

$f^{(k)}(x) = g(x) + \lambda \int_a^x K(x,s) f(s) ds$

where,

*   $f^{(k)}(x)$ denotes the k-th derivative of $f(x)$  i.e. $f^{(k)}(x) = \frac{d^k}{dx^k}f(x)$,   $k = 1,2,...,n$

* $g(x)$ is some given arbitrary function,

* $\lambda, a$ are constant parameters,

*   and , $K(x,s)$ is the kernel function that typically encodes the relationship between the present and past states.

The Volterra equation reflects the dependence of $f^{(k)}(x)$ on the history of the system through the kernel $K(x,s)$, which captures how past values $\{u(s)\}$ influence the present value at time instant $t$. This can be used particularly in situations where the system's behavior is non-Markovian in nature. Here we shall adopt it to model the dynamics of a *semiconductor quantum-dot biexciton (3-level system) strongly coupled to an acoustic-phonon environment*.

<br>

### The Nakajima-Zwanzig equation

The **Nakajima-Zwanzig** (NZ) equation is a non-Markovian master equation commonly used in the study of open quantum systems to describe the evolution of a system’s reduced density matrix while accounting for the system's interactions with its environment that builds up non-local memory effects.

Given a biexciton system with the ground, exciton and the biexciton states represented as $|0\rangle, |1\rangle, |2\rangle$ respectively. The EOM of the reduced density matrix $\rho$ of this system is described by the NZ equation as,

$\dot\rho(t) = -i[\hat{H}_S(t),\rho(t)] + \int_0^t \hat{\mathcal{K}}(t,t') \rho(t') dt'$

where,

*   system Hamiltonian: $\hat{H}_S = \begin{pmatrix}
\Delta & -\Omega/2 & 0 \\
-\Omega/2 & -E_b/2 & -\Omega/2 \\
0 & -\Omega/2 & -\Delta
\end{pmatrix}$

*   coupling matrix: $\hat{S} = \begin{pmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 2
\end{pmatrix}$

* $\hat{\mathcal{K}}(t,t') \rho(t') = i[\hat{S}, \{i c(t,t')\hat{U}(t,t')\hat{S}\rho(t')\} + \{H.c.\}]$

    *  time-ordered unitary propagator : $\hat{U} = \mathcal{T} e^{-i\int_{t'}^t\hat{H}_S dt''}$

    *   correlation function : $c(t,t') = \int_{-\infty}^{+\infty} n_\beta (\omega) J(\omega) e^{i \omega (t-t')}d\omega$

        *   phonon occupation number : $n_\beta(\omega) = 1/(e^{\beta\omega} -1)$

        *   inverse temperature : $\beta = \hbar/k_B T$

        *   spectral density function : $J(\omega) = A \omega^3 e^{-\omega^2/\omega_c^2}$

<br><br>

## Goal

A biexciton QD-system is acted upon by a Hamiltonian $\hat{H}_S$ (using a constant $\pi$-pulse and no detuning) and evolves in the presence of phonon-induced decoherence as dictated by the NZ equation. Such a setup is developed to efficiently transfer the population from the ground state to the biexciton state. *Can a PINN learn the solution to this EOM i.e. $\rho(t)$ using only the knowledge of NZ equation and the initial boundary conditions?*

<br>

## Workflow

We shall implement and train a fully connected dense neural network that directly approximates the solution to the Volterra equation i.e. $NN(\textbf{w}, t) \sim \rho(t)$. Note that the density matrix $\rho$ is a $3\times3$ complex valued matrix and Pytorch can't handle complex valued outputs in the usual manner as real values. So we shall address this issue in the following way.

<br>

## Architecture

The exciton and biexciton model are designed differently. The general idea is that the neural network takes time instance $t$ as the input and output elements of the density matrix of the system. For the exciton model 3 outputs suffices while for the biexciton model 8 outputs are required as mentioned below.

*   Input : $t \in ℝ$
*   Output : $NN_{ex}(\textbf{w}, t) =  \bigg[ \rho_{00}^{\text{ex}}(t_i),  Re[\rho_{01}^{\text{ex}}(t_i)],  Im[\rho_{01}^{\text{ex}}(t_i)]\bigg]   \in ℝ^3$

$\hspace{2cm} NN_{biex}(\textbf{w}, t) = \bigg[ \rho_{00}^{\text{biex}}(t_i),  \rho_{11}^{\text{biex}}(t_i),  Re[\rho_{01}^{\text{biex}}(t_i)],  Im[\rho_{01}^{\text{biex}}(t_i)], Re[\rho_{02}^{\text{biex}}(t_i)], Im[\rho_{02}^{\text{biex}}(t_i)],  $
$\hspace{10cm} Re[\rho_{12}^{\text{biex}}(t_i)],  Im[\rho_{12}^{\text{biex}}(t_i)] \bigg]   \in  ℝ^8$

The complete density matrix for the exciton is constructed as follows

*   $\rho_{01} = Re[\rho_{01}] + Im[\rho_{01}]$

*   $\rho_{10} = \rho_{01}^*$

*   $\rho_{11} = 1 - \rho_{00}$

*   $\rho = [[\rho_{00}, \rho_{01}], [\rho_{10},\rho_{11}]]$

while for the biexciton,

*   $\rho_{01} = Re[\rho_{01}] + Im[\rho_{01}]$

*   $\rho_{02} = Re[\rho_{02}] + Im[\rho_{02}]$

*   $\rho_{12} = Re[\rho_{12}] + Im[\rho_{12}]$

*   $\rho_{10} = \rho_{01}^*$

*   $\rho_{20} = \rho_{02}^*$

*   $\rho_{21} = \rho_{12}^*$

*   $\rho_{22} = 1 - \rho_{00} - \rho_{11}$

*   $\rho = [[\rho_{00}, \rho_{01}, \rho_{02}], [\rho_{10},\rho_{11}, \rho_{12}], [\rho_{20}, \rho_{21}, \rho_{22}]]$


Using *torch.autograd's grad* function we can calculate the gradient of the density matrix. But the real and imaginary parts need to be adressed separately and then combined again as before.
<br>

## Loss function

The total loss consists of 2 components:

$\mathcal{L} = \mathcal{L}_{boundary} + \lambda \mathcal{L}_{physics} $

*   $\mathcal{L}_{boundary}$ : penalizes the initial boundary conditions to  $\rho^*(0) = diag(1,0,0)$ as,

    $min \ ||\rho(NN(\textbf{w}, 0)) - \rho^*(0) ||^2$

*   $\mathcal{L}_{physics}$ : penalizes the Volterra equation i.e.

    $min \ \sum_{t \in [0,D]}||\dot\rho(NN(\textbf{w},t)) + i[\hat{H}_S(t),\rho(NN(\textbf{w},t))] - \int_0^t \hat{\mathcal{K}}(t,t') \rho(NN(\textbf{w},t')) dt' ||^2 $


where, $||A_{3\times3}||^2 = \sum_{i,j} A_{ij}^2 / 9$