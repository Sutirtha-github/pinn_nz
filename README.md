# **Learning the Nakajima-Zwanzig Equation Of Motion using Physics Informed Neural Network**

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

Let the neural network take time instance $t$ as the input and output the wavefunction $\psi(t)$ which is a vector of shape $3\times1$. But in order to deal with the complex probability amplitudes of $\psi$, let the neural network output the real and imaginary parts of each of these 3 amplitudes separately, thereby producing 6 outputs in total.

*   Input : $t \in ℝ$
*   Output : $NN(\textbf{w}, t) \in ℝ^6$

If $\psi(t) = \begin{pmatrix}
c_0(t) \\
c_1(t) \\
c_2(t)
\end{pmatrix}, \{c_0, c_1, c_2\} \in ℂ$

then,

*   $NN(\textbf{w}, t)[0] = Re[c_0(t)], \hspace{1cm} NN(\textbf{w}, t)[3] = Im[c_0(t)]$

*   $NN(\textbf{w}, t)[1] = Re[c_1(t)], \hspace{1cm} NN(\textbf{w}, t)[4] = Im[c_1(t)]$

*   $NN(\textbf{w}, t)[2] = Re[c_2(t)], \hspace{1cm} NN(\textbf{w}, t)[5] = Im[c_2(t)]$

From these 6 outputs we can construct $\psi$ as,

$\psi(NN(\textbf{w},t)) = \begin{pmatrix}
NN(\textbf{w}, t)[0] + i\ NN(\textbf{w}, t)[3]\\
NN(\textbf{w}, t)[1] + i\ NN(\textbf{w}, t)[4]\\
NN(\textbf{w}, t)[1] + i\ NN(\textbf{w}, t)[5]
\end{pmatrix}$

and then finally obtain the density matrix,

$\rho(NN(\textbf{w},t)) = |\psi(NN(\textbf{w},t))\rangle\langle\psi(NN(\textbf{w},t))|$

Using *torch.autograd's grad* function we can calculate the gradient of the density matrix.

$\dot{\rho}(NN(\textbf{w},t)) = |\dot{\psi}\rangle\langle\psi| + |\psi\rangle\langle\dot{\psi}|$

<br>

## Loss function

The total loss consists of 3 components:

$\mathcal{L} = \mathcal{L}_{boundary} + \mathcal{L}_{physics} + \mathcal{L}_{norm}$

*   $\mathcal{L}_{boundary}$ : penalizes the initial boundary conditions to  $\rho^*(0) = diag(1,0,0)$ as,

    $min \ ||\rho(NN(\textbf{w}, 0)) - \rho^*(0) ||^2$

*   $\mathcal{L}_{physics}$ : penalizes the Volterra equation i.e.

    $min \ \sum_{t \in [0,D]}||\dot\rho(NN(\textbf{w},t)) + i[\hat{H}_S(t),\rho(NN(\textbf{w},t))] - \int_0^t \hat{\mathcal{K}}(t,t') \rho(NN(\textbf{w},t')) dt' ||^2 / N$

*   $\mathcal{L}_{norm}$ : penalizes the trace of the density matrix to ensure normalization of the wavefunction (sum of populations = 1) .

    $min \ \sum_{t \in [0,D]}||Tr(\rho(NN(\textbf{w}, t)) - 1.0 ||^2 / N$

where, $||A_{3\times3}||^2 = \sum_{i,j} A_{ij}^2 / 9$