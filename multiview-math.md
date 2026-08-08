# Multiview Registration by Synchronization — Mathematical Reference

The math behind `teaser/src/rotation_sync.cc`, `teaser/src/translation_sync.cc` and the driver in
`teaser/src/multiview.cc`.

**Notation.** 
- $V$ is the set of scans
- $E$ the set of overlapping pairs. 
- Scan $i$ has an unknown
pose $(R_i, t_i) \in \mathrm{SE}(3)$ mapping its local frame to the world, $w = R_i p + t_i$. Edge
$(p,q)$ carries correspondences $j \in \mathrm{Corr}(p,q)$, each a pair $(a_j, b_j)$ of points
expressed in $p$'s and $q$'s local frames respectively. 
- Scale is fixed at $1$ throughout.

---

## 1. The optimization problem

Two scans see the same physical point, so at the true poses

$$R_p a_j + t_p \;=\; R_q b_j + t_q .$$

Correspondences come from feature matching, so a large fraction are simply wrong. So **Truncated Least Squares** is used:

$$
\operatorname*{arg\,min}_{\{R_i,\, t_i\}}
\sum_{(p,q) \in E} \; \sum_{j \in \mathrm{Corr}(p,q)}
\min\!\left(
\frac{\bigl\lVert (R_p a_j + t_p) - (R_q b_j + t_q) \bigr\rVert^2}{\sigma_j^2},
\; 1 \right)
\tag{1}
$$

subject to $R_i \in \mathrm{SO}(3)$, where $\sigma_j$ is the noise bound on that residual (§6).

### Gauge freedom

The objective depends only on relative poses. For any $Q \in \mathrm{SO}(3)$ and
$\tau \in \mathbb{R}^3$, the substitution

$$R_i \;\leftarrow\; Q R_i, \qquad t_i \;\leftarrow\; Q t_i + \tau$$

leaves every residual unchanged. So $(1)$ determines the poses only up to a **global rigid
transform, independently per connected component** of $E$. Here each component's lowest-indexed node is pinned to $(I, 0)$.



| Difficulty | Mechanism | Section |
|---|---|---|
| $\min(\cdot, 1)$ is non-convex and combinatorial | Graduated Non-Convexity | §2 |
| $R$ and $t$ coupled over a graph; $\mathrm{SO}(3)$ non-convex | TIM decoupling + spectral relaxation | §3–§5 |

---

## 2. Graduated Non-Convexity: from TLS to weighted least squares

Write $\hat r_j = r_j / \sigma_j$ for the normalized residual. The Black–Rangarajan duality
re-expresses the truncated cost through a **line process** $w_j \in [0,1]$:

$$
\min\bigl(\hat r_j^{\,2},\, 1\bigr)
\;=\;
\min_{w_j \in [0,1]} \; w_j\, \hat r_j^{\,2} + (1 - w_j)
\tag{2}
$$

the minimizer picking $w_j = 1$ (keep) when $\hat r_j^{\,2} < 1$ and $w_j = 0$ (discard) otherwise.
This converts $(1)$ into a joint minimization over poses **and** weights, but does not make it
convex.

GNC replaces the penalty by a family $\Phi_\mu$ controlled by $\mu > 0$, chosen so that
$\mu \to 0$ gives a near-quadratic (convex, all-inliers) surrogate and $\mu \to \infty$ recovers the
true truncated cost. The resulting problem is solved by alternating two steps, each closed-form.

**Fix $w$, solve for the poses.** The cost becomes a *weighted least squares* problem — everything
in §3–§5 is this step.

**Fix the poses, solve for $w$.** Independently per residual,

$$
w_j =
\begin{cases}
0
  & \text{if } \hat r_j^{\,2} \;\ge\; \dfrac{\mu+1}{\mu} \quad (\text{outlier}) \\[3mm]
1
  & \text{if } \hat r_j^{\,2} \;\le\; \dfrac{\mu}{\mu+1} \quad (\text{inlier}) \\[3mm]
\sqrt{\dfrac{\mu(\mu+1)}{\hat r_j^{\,2}}} \;-\; \mu
  & \text{otherwise} \quad (\text{undecided})
\end{cases}
\tag{3}
$$

**Annealing.** $\mu$ is initialized so that no residual starts outside the undecided band,

$$\mu_0 \;=\; \frac{1}{2 \max_j \hat r_j^{\,2} - 1}\tag{4}$$

(if this is $\le 0$ the data already lies within its noise bound and there is nothing to anneal),
then $\mu \leftarrow \gamma\mu$ each iteration with $\gamma > 1$. As $\mu$ grows the two thresholds
in $(3)$ converge on $1$ and the surrogate sharpens into the true truncated cost. Iteration stops
when the change in cost falls below a threshold.

Normalizing residuals by $\sigma_j$ up front is what makes $(3)$ and $(4)$ free of $\sigma$: the
noise bound is $1$ by construction, so a single weight-update routine serves both the rotation and
translation stages even though their noise bounds differ (§6).

---

## 3. Decoupling rotation from translation

Rotation and translation are coupled in $(1)$. The decoupling is a change of measurement.

### Translation-Invariant Measurements

Take two correspondences $j, k$ on the same edge and subtract their equations:

$$(R_p a_j + t_p) - (R_p a_k + t_p) \;=\; R_p (a_j - a_k)$$

and likewise for $q$. The translations cancel identically. Defining the **TIMs**

$$x_{jk} = a_j - a_k \quad (\text{in } p\text{'s frame}), \qquad
  y_{jk} = b_j - b_k \quad (\text{in } q\text{'s frame}),$$

the constraint becomes

$$R_p\, x_{jk} \;=\; R_q\, y_{jk}\tag{5}$$

which involves **no translation at all**. So the rotations can be estimated first, from TIMs alone;
the translations then follow from a linear problem (§5).



### The TIM graph

$(5)$ holds for *every* pair $(j,k)$, but the algorithm does not use every pair. A TIM is picked per
edge of a graph whose **vertices are the correspondences** of $(p,q)$, and that graph is a design
choice. With $K$ correspondences on the edge, TEASER offers two
(`INLIER_GRAPH_FORMULATION`, `registration.h:417`):

| Formulation | TIM count | Each point appears in | Built by |
|---|---|---|---|
| `COMPLETE` | $K(K-1)/2$ — all pairs | $K-1$ TIMs | `computeTIMs`, `registration.cc` |
| `CHAIN` | $K$ — a single cycle | $2$ TIMs | inline loop, `registration.cc:717–748` |

`CHAIN` takes consecutive differences over the (sorted) inlier list and wraps the last back to the
first, so the TIMs form one cycle rather than a clique.

Alongside the vectors, `computeTIMs` returns a $2 \times K(K-1)/2$ **index map** recording which
correspondence pair each TIM came from. That map is what makes the pipeline work: it converts
TIM-level consistency tests back into a graph over *correspondences*, whose maximum clique is the
inlier set (§8). Without it a rejected TIM could not be attributed to the points responsible.

TIMs are computed **twice**, for different purposes:

1. **Before pruning**, on all $K$ correspondences — always `COMPLETE`. These feed the
   scale-consistency test and the max-clique inlier graph, which need every pair to be testable.
2. **After pruning**, on the $K'$ clique members — these are the rotation stage's input.
   The pairwise pipeline makes this one selectable, defaulting to **`CHAIN`**
   (`registration.h:478`).

The multiview path calls `computeTIMs` for both (`multiview.cc:297, 298, 315, 316`), so it is
`COMPLETE` throughout and currently **ignores `params.rotation_tim_graph`**. That is a deliberate
consequence of solving jointly — with $|E|$ edges each contributing a spectral summary rather than
one edge contributing everything, more constraints per edge cost little — but it is a divergence from
the pairwise default worth knowing.

### Cost of the decoupling

The rotation cost $(6)$ sums over **TIMs, not correspondences**, so the effective number of terms is
the TIM count above — quadratic in $K$ under `COMPLETE`. For a 30-member clique that is 435 terms
per edge, per GNC iteration.

TIMs are differences of noisy measurements, so their noise bound is the sum of their endpoints' (§6),
and TIMs sharing an endpoint have correlated errors. 

Either way the decoupling trades statistical efficiency for two things: the removal of $t$ from the
rotation problem, and — more valuable in practice — the ability to reject outliers *before* rotation,
because $(5)$ is testable on a single pair without knowing any pose.

---

## 4. Rotation

The weighted least-squares step of §2, restricted to the rotation subproblem, is

$$
\operatorname*{arg\,min}_{\{R_i\}} \;
\sum_{(p,q) \in E} \; \sum_j \; w_j \bigl\lVert R_p x_j - R_q y_j \bigr\rVert^2
\tag{6}
$$

over $R_i \in \mathrm{SO}(3)$. 

### 4.1 Per-edge reduction

Expand the summand for one edge. Since $R$ is orthogonal, $\lVert R_p x\rVert = \lVert x\rVert$, so

$$
\sum_j w_j \lVert R_p x_j - R_q y_j \rVert^2
=
\sum_j w_j \bigl( \lVert x_j\rVert^2 + \lVert y_j\rVert^2 \bigr)
\;-\; 2 \sum_j w_j\, x_j^\top R_p^\top R_q\, y_j .
$$

The first group is constant in the rotations. With the **weighted cross-covariance**

$$M_{pq} \;=\; \sum_j w_j\, x_j y_j^\top \;=\; X \operatorname{diag}(w) Y^\top \;\in\; \mathbb{R}^{3\times 3}\tag{7}$$

minimizing $(6)$ on this edge is equivalent to maximizing $\operatorname{tr}(M_{pq} A)$ over the
relative rotation $A = R_q^\top R_p$. Writing the SVD $M_{pq} = U\Sigma V^\top$ and
$Z = V^\top A U$ (orthogonal),

$$
\operatorname{tr}(M A)
= \operatorname{tr}\bigl(U \Sigma V^\top A\bigr)
= \operatorname{tr}(\Sigma Z)
= \sum_i \sigma_i Z_{ii}
\;\le\; \sum_i \sigma_i
$$

with equality at $Z = I$, hence

$$\hat A_{pq} \;=\; V \operatorname{diag}\!\bigl(1,\, 1,\, \det(V U^\top)\bigr) U^\top
\;\approx\; R_q^\top R_p .\tag{8}$$

The $\det$ factor projects onto the proper-rotation branch: $VU^\top$ may be a reflection, and
flipping the **last** column is the cheapest correction because singular values are ordered
descending, so the sacrificed axis is the least-supported one. Note the orientation carefully —
$V U^\top$, not $U V^\top$; the two are transposes and only one satisfies $(11)$ below.

> **No centering.** Classical Procrustes subtracts centroids before forming $M$. Here TIMs are
> already translation-free, so $M$ is formed from the vectors as given.

Each edge is thus summarized by a rotation $\hat A_{pq}$ and a scalar **confidence** $c_{pq} > 0$.



### 4.2 The synchronization objective

Stack the unknowns as a $3n \times 3$ matrix $\rho$ with blocks $\rho_p = R_p^\top$, and assemble
the symmetric block matrix $B \in \mathbb{R}^{3n \times 3n}$:

$$B_{[q][p]} = c_{pq} \hat A_{pq}, \qquad
  B_{[p][q]} = c_{pq} \hat A_{pq}^\top, \qquad
  B_{[p][p]} = 0 .\tag{9}$$

Using $\bigl\lVert R_q^\top R_p - \hat A_{pq} \bigr\rVert_F^2
= 6 - 2\operatorname{tr}\bigl(\hat A_{pq}^\top R_q^\top R_p\bigr)$, minimizing the total discrepancy
between each edge's measured and realized relative rotation is equivalent to

$$\max_{\rho} \; \operatorname{tr}\bigl(\rho^\top B \rho\bigr)
\qquad \text{subject to} \qquad \rho_p \in \mathbf{SO}(3) \;\; \forall p .\tag{10}$$

This is still non-convex — the constraint set is a product of $\mathrm{SO}(3)$s — and NP-hard in
general.

### 4.3 Spectral relaxation

Define the block-diagonal degree matrix $D$ with $D_{[p][p]} = d_p I_3$ and
$d_p = \sum_{q \sim p} c_{pq}$. On **exact** data, where $\hat A_{pq} = R_q^\top R_p$ holds
identically, the truth is an eigenvector:

$$(B\rho)_q
= \sum_p c_{pq} \bigl(R_q^\top R_p\bigr) R_p^\top
= \Bigl( \sum_p c_{pq} \Bigr) R_q^\top
= d_q\, \rho_q\tag{11}$$

so $B\rho = D\rho$ — a generalized eigenvalue problem with eigenvalue $1$ and an eigenspace of
dimension $3$, one per column, i.e. one per gauge degree of freedom. Substituting
$\rho = D^{-1/2} z$ gives the ordinary symmetric problem

$$B_n \;=\; D^{-1/2} B D^{-1/2}, \qquad B_n z = z .\tag{12}$$

$B_n$ has spectral radius $\le 1$: each block row of $D^{-1}B$ is a convex combination of rotation
matrices, whose operator norm is at most $1$. So the truth attains the **largest** eigenvalue.

The relaxation replaces the constraint $\rho_p \in \mathrm{SO}(3)$ by the far weaker
$\rho^\top \rho = I$ (a Stiefel manifold), for which $(10)$ is maximized by the top-3 eigenvectors
of $B_n$. Undoing the normalization,

$$Y \;=\; D^{-1/2} Z_{1:3}, \qquad\text{blocks}\qquad Y_p \;\approx\; \nu_p\, R_p^\top Q\tag{13}$$

for an unknown common $Q \in \mathrm{O}(3)$ (the gauge) and positive per-node scalars $\nu_p$.

**Conditioning.** The relaxation is tight when the top-3 eigenspace is well separated, so
$\mathrm{gap} = \lambda_3 - \lambda_4$ is the natural diagnostic. A small gap means the graph is
weakly connected and the recovered rotations are poorly determined — add edges rather than iterate
harder.

### 4.4 Rounding and reflection

Each block $Y_p$ is a scaled, noisy rotation. Project it back with the nearest-rotation map — a
*different* problem from $(8)$, namely
$\operatorname*{arg\,min}_{R \in \mathrm{SO}(3)} \lVert R - Y_p \rVert_F
= \operatorname*{arg\,max} \operatorname{tr}(R^\top Y_p)$:

$$Y_p = U \Sigma V^\top
\quad\Longrightarrow\quad
\tilde R_p = U \operatorname{diag}\!\bigl(1,\,1,\,\det(U V^\top)\bigr) V^\top,
\qquad R_p = \tilde R_p^\top\tag{14}$$

Note $U V^\top$ here, the transpose of the orientation in $(8)$; the final transpose is because the
blocks of $Y$ carry $R_p^\top$.

The gauge $Q$ in $(13)$ lies in $\mathrm{O}(3)$, not $\mathrm{SO}(3)$: it may be a reflection, in
which case **every** block has negative determinant. Detect it by the sign of
$\sum_p \operatorname{sign}\det Y_p$ and, if negative, flip one column of $Y$ — which flips all
blocks simultaneously. A node whose determinant sign then still disagrees with its component is
genuinely inconsistent with the rest of the graph, and is flagged.

### 4.5 Gauge

The rounded rotations are determined up to the left action $R_p \leftarrow Q R_p$. Pinning the
anchor $a$,

$$R_p \;\leftarrow\; R_a^\top R_p \qquad (\text{so } R_a = I \text{ exactly})\tag{15}$$

This is cosmetic for the cost but makes the output deterministic across runs rather than drifting
with whatever basis the eigensolver returned.

---

## 5. Translation

With the rotations fixed, the weighted least-squares step of $(1)$ in the translations is **linear**:

$$
\operatorname*{arg\,min}_{\{t_i\}} \;
\sum_{(p,q) \in E} \; \sum_j \; w_j
\bigl\lVert (R_p a_j + t_p) - (R_q b_j + t_q) \bigr\rVert^2
\tag{16}
$$

### 5.1 The per-edge reduction is exact

Put $u = t_p - t_q$ and $c_j = R_q b_j - R_p a_j$, so the residual is $u - c_j$. With
$W_{pq} = \sum_j w_j$ and the weighted mean $d_{pq} = \bigl(\sum_j w_j c_j\bigr) / W_{pq}$,

$$
\sum_j w_j \lVert u - c_j \rVert^2
= W_{pq}\lVert u \rVert^2 - 2 W_{pq}\, u^\top d_{pq} + \sum_j w_j \lVert c_j \rVert^2
= W_{pq} \bigl\lVert u - d_{pq} \bigr\rVert^2
+ \underbrace{\Bigl( \sum_j w_j \lVert c_j\rVert^2 - W_{pq}\lVert d_{pq}\rVert^2 \Bigr)}_{\text{independent of } u}
\tag{17}
$$

So collapsing an edge's $m$ correspondences to the single target

$$d_{pq} \;=\; R_q\, \bar b_{pq} \;-\; R_p\, \bar a_{pq}
\qquad (\bar a, \bar b \text{ the weighted centroids of that edge's points})\tag{18}$$

loses nothing — it is an exact reformulation, not an approximation. Note $d_{qp} = -d_{pq}$.

It does discard the per-correspondence residuals, so a GNC loop must retain the raw points to
rebuild $r_j = (t_p - t_q) - c_j$ for the weight update $(3)$.

### 5.2 Normal equations are a graph Laplacian

The reduced problem is
$\min \sum_{(p,q)} W_{pq} \lVert t_p - t_q - d_{pq} \rVert^2$. Setting the gradient at node $p$ to
zero,

$$\sum_{q \sim p} W_{pq}\,(t_p - t_q) \;=\; \sum_{q \sim p} W_{pq}\, d_{pq} .\tag{19}$$

The left side is exactly the **weighted graph Laplacian** applied to the translations, so stacking
the nodes as rows of $T \in \mathbb{R}^{n \times 3}$,

$$L\,T = \mathcal{B},
\qquad
L_{pp} = \sum_{q\sim p} W_{pq},
\quad
L_{pq} = -W_{pq},
\quad
\mathcal{B}_{[p]} = \sum_{q \sim p} W_{pq}\, d_{pq}^\top .\tag{20}$$

Three right-hand sides — $x$, $y$, $z$ — share one matrix, so one factorization solves all three.

### 5.3 Singularity is the gauge, and consistency is automatic

$L \mathbf{1} = 0$: the constant shift is precisely the translation gauge of §1, so $L$ is singular
with a one-dimensional nullspace per connected component. The system is nonetheless consistent,
because the antisymmetry $d_{qp} = -d_{pq}$ gives

$$\mathbf{1}^\top \mathcal{B}
= \sum_{(p,q) \in E} \bigl( W_{pq} d_{pq}^\top - W_{pq} d_{pq}^\top \bigr)
= 0\tag{21}$$

so $\mathcal{B}$ lies in the range of $L$. Fixing the gauge by pinning the anchor $t_a = 0$ and
deleting row and column $a$ yields the **grounded Laplacian**, which is positive definite for a
connected component and stays sparse — so a sparse Cholesky factorization applies directly.

> The textbook alternative, factorizing $L + \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$ to impose
> $\sum_p t_p = 0$, is both denser (that rank-one term is full) and pointless here, since anchoring
> discards its zero-mean gauge anyway.

### 5.4 Uncertainty

Because $(20)$ is linear, uncertainty propagates in closed form. The **effective resistance**
between two nodes,

$$R_{\mathrm{eff}}(p,q) \;=\; (e_p - e_q)^\top L^{+} (e_p - e_q),\tag{22}$$

is the variance amplification factor of the relative translation $t_p - t_q$; multiplied by the
residual scale it gives that relative translation's standard deviation. It is obtained by solving
$L z = e_p - e_q$ with the existing factorization, never by forming $L^{+}$. At the graph level,
$\lambda_2(L)$ (the Fiedler value) plays the role the spectral gap plays for rotation: small means a
bottleneck, and translations across it are poorly determined.

---

## 6. Noise bounds

A per-point bound $\delta_i$ on scan $i$ composes differently for the two stages, because they
consume different measurements. These are hard bounds, so they add by the triangle inequality — not
in quadrature.

| Stage | Measurement | Composition | Uniform $\delta$ |
|---|---|---|---|
| Rotation | TIM $\leftrightarrow$ TIM | $\sigma_{pq} = 2\delta_p + 2\delta_q$ | $4\delta$ |
| Translation | point $\leftrightarrow$ point | $\sigma_{pq} = \delta_p + \delta_q$ | $2\delta$ |

A TIM is a difference of two points from the same scan, so its own bound is $2\delta$; both sides of
a rotation residual are TIMs, and both scans are noisy, so the two contributions add. A translation
residual compares raw points, one from each scan, so each contributes once.

This is the **symmetric** composition, appropriate because every pose here is unknown. It differs
from pairwise TEASER, whose measurement model treats the source cloud as exact and attributes all
noise to the destination — hence its factor of $2$ where this has $4$. Reusing the pairwise factor
would make the bound too tight by $2\times$ when both scans are equally noisy.

$\sigma_{pq}$ enters in **two distinct roles**, and both are needed:

1. **Information weight** — how hard an edge pulls: $c_{pq} \propto 1/\sigma_{pq}^2$ for rotation,
   and $W_{pq} \propto 1/\sigma_{pq}^2$ for translation.
2. **Rejection threshold** — what counts as an outlier: via the normalization
   $\hat r = r/\sigma$ in $(3)$.

Handling only the first gives correct relative influence but classifies outliers against the wrong
bound; handling only the second classifies correctly but lets a $10\times$ noisier correspondence
pull as hard as a precise one.

---

## 7. Invariances

Useful for reasoning about the algorithm, and each one a trap if forgotten.

| Transformation | Effect | Why |
|---|---|---|
| $R_i \leftarrow Q R_i$, $t_i \leftarrow Q t_i + \tau$ | none | gauge freedom, §1 |
| Scale one edge's $x, y$ by $\alpha > 0$ | none on rotation | $\operatorname{polar}(\alpha^2 M) = \operatorname{polar}(M)$, §4.1 |
| Scale one edge's $w$ by $\alpha > 0$ | none on rotation | same |
| Scale **all** $c_{pq}$ by $\alpha > 0$ | none | cancels in $D^{-1/2} B D^{-1/2}$ |
| Scale **all** $W_{pq}$ by $\alpha > 0$ | none on translation | scales $L$ and $\mathcal{B}$ equally in $(20)$ |
| Uniform noise bounds $\delta_i \equiv \delta$ | none on either weighting | a common factor on all weights |
| Recenter cloud $p$ by $o_p$ | none on translation | shifts $d_{pq}$ by $R_p o_p - R_q o_q$, which the undo $t_p \leftarrow t_p - R_p o_p$ cancels exactly |

The third row is the sharp one: per-correspondence weights cannot, by themselves, suppress an edge
that is *coherently* wrong. Only $c_{pq}$ can.

---

## 8. The algorithm end to end

**Per edge $(p,q)$ — prune once, up front.**

1. Build the point pairs and form their TIMs.
2. Scale-consistency test $\bigl\lvert\, \lVert y_k \rVert - \lVert x_k \rVert \,\bigr\rvert \le \sigma_{pq}$ on every TIM, giving a graph over correspondences.
3. Take the maximum clique of that graph; keep only those correspondences.

The test in step 2 is the pairwise-testable consequence of $(5)$: a rigid transform preserves TIM
length, so two correspondences whose TIM lengths disagree by more than the noise bound cannot both
be inliers. The maximum clique is the largest mutually consistent subset, and pruning to it before
the GNC loops removes gross outliers cheaply.

**Rotation, GNC loop.** Repeat until the cost converges:

1. $R \leftarrow$ spectral synchronization (§4) with the current weights $w$.
2. $\hat r_j^{\,2} \leftarrow \lVert R_p x_j - R_q y_j \rVert^2 / \sigma_{\mathrm{rot}}^2$.
3. $w \leftarrow$ weight update $(3)$; then $\mu \leftarrow \gamma\mu$.

**Translation, GNC loop.** Repeat until the cost converges:

1. $t \leftarrow$ Laplacian solve (§5) given $R$, with the current weights $w$.
2. $\hat r_j^{\,2} \leftarrow \lVert (R_p a_j + t_p) - (R_q b_j + t_q) \rVert^2 / \sigma_{\mathrm{trans}}^2$.
3. $w \leftarrow$ weight update $(3)$; then $\mu \leftarrow \gamma\mu$.

**Finally**, re-gauge each component to its anchor (§4.5, §5.3).

Where each piece lives:

| Section | Code |
|---|---|
| §2 GNC weight update, annealing | `multiview.cc` (`gncUpdateWeights`, `gncStep`) |
| §4.1 per-edge reduction | `rotation_sync.cc`, via `utils::svdRot` |
| §4.2–4.5 spectral solve | `rotation_sync.cc` (`synchronizeRotations`) |
| §5 Laplacian solve, diagnostics | `translation_sync.cc` (`synchronizeTranslations`) |
| §6 bound composition | `timResidualNoiseBound`, `pointResidualNoiseBound` |
| §8 pruning, driver | `multiview.cc` (`alignMultiScan`, `alignMultiScanWithGraph`) |

---

## References

- H. Yang, J. Shi, L. Carlone, *TEASER: Fast and Certifiable Point Cloud Registration* — TIM
  decoupling, scale-consistency pruning, max-clique inlier selection.
- H. Yang, P. Antonante, V. Tzoumas, L. Carlone, *Graduated Non-Convexity for Robust Spatial
  Perception* — the surrogate and weight update $(3)$.
- M. J. Black, A. Rangarajan, *On the Unification of Line Processes, Outlier Rejection, and Robust
  Statistics* — the duality $(2)$.
- A. Singer, *Angular Synchronization by Eigenvectors and Semidefinite Programming* — the spectral
  relaxation of §4.3.
- F. Arrigoni, A. Fusiello, *Synchronization Problems in Computer Vision with Closed-Form
  Solutions* — survey of rotation and translation synchronization.
