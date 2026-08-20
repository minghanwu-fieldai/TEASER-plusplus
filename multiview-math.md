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
$V U^\top$, not $U V^\top$; the two are transposes and only one satisfies $(13)$ below.

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
\qquad \text{subject to} \qquad \rho_p \in \mathrm{SO}(3) \;\; \forall p .\tag{10}$$

This is still non-convex — the constraint set is a product of $\mathrm{SO}(3)$s — and NP-hard in
general.

**The dropped "constant" is only constant while the blocks are orthogonal.** That $6 = 3 + 3$ used
$\lVert R_q^\top R_p\rVert_F^2 = 3$, which fails the moment $\rho_p$ is relaxed off $\mathrm{SO}(3)$.
So for the relaxation, write the same weighted discrepancy in the form that survives — note
$\hat A_{pq}^\top \rho_q = R_p^\top R_q R_q^\top = \rho_p$ at the truth:

$$\sum_{(p,q) \in E} c_{pq}\bigl\lVert \rho_p - \hat A_{pq}^\top \rho_q \bigr\rVert_F^2
\;=\; \operatorname{tr}\!\bigl(\rho^\top (D - B)\,\rho\bigr),
\qquad D_{[p][p]} = d_p I_3,\quad d_p = \sum_{q \sim p} c_{pq}
\tag{10a}$$

Expanding gives $\sum_p d_p\lVert\rho_p\rVert_F^2 = \operatorname{tr}(\rho^\top\! D\rho)$ from the
squared terms and $\operatorname{tr}(\rho^\top B\rho)$ from the cross terms; it uses only that
$\hat A_{pq}$ is orthogonal, **not** that $\rho_p$ is. $D - B$ is the **connection Laplacian**, the
rotational analogue of the graph Laplacian $D - A$ that appears for translation in §5.2.

Two consequences set up everything below:

- **The truth is in the null space.** At the truth every edge residual in $(10a)$ vanishes, so
  $(D-B)\rho = 0$ — exactly as $L\mathbf 1 = 0$ in §5.3. "Singularity is the gauge" holds for
  rotation too.
- **$\operatorname{tr}(\rho^\top\! D\rho)$ becomes a live variable.** Once blocks may shrink, the
  cost can be lowered by shrinking $\lVert\rho_p\rVert$ instead of aligning anything. Pinning that
  term is precisely what the relaxation's constraint has to do — which is why the constraint is
  $D$-weighted.

### 4.3 Spectral relaxation

**The relaxed problem.** Block-orthogonality $\rho_p^\top\rho_p = I_3$ for every $p$ implies the
aggregate statement $\rho^\top\! D\rho = \bigl(\sum_p d_p\bigr) I_3$ — *isotropic*. Keep only that
consequence and drop the per-block constraint:

$$\min_{\rho} \; \operatorname{tr}\!\bigl(\rho^\top (D-B)\rho\bigr)
\qquad \text{subject to} \qquad \rho^\top\! D\, \rho \;\propto\; I_3 .\tag{11}$$

Only the *shape* is the constraint. The magnitude is a convention: the objective is homogeneous of
degree 2, so $\rho^\top\!D\rho = I$ and $\rho^\top\!D\rho = (\sum_p d_p)I$ differ by one uniform
scale factor and give the same column span — and hence, since rounding $(15)$ is scale-invariant, the
same rotations. Fixing it to $I$ puts $z = D^{1/2}\rho$ on the standard Stiefel manifold:

$$B_n \;=\; D^{-1/2} B D^{-1/2},
\qquad
\min \operatorname{tr}\!\bigl(z^\top (I - B_n) z\bigr)
\;\Longleftrightarrow\;
\max \operatorname{tr}\!\bigl(z^\top B_n z\bigr),
\quad z^\top z = I
\tag{12}$$

so $\rho = D^{-1/2} Z$ with $Z$ the **top-3 eigenvectors of $B_n$**. Here $I - B_n$ is the
*normalized* connection Laplacian, whose bottom-3 eigenvectors are $B_n$'s top-3 — SGHR writes
$D-B$ and takes the three smallest, which is the same computation in the other variable.

> **Not plain Stiefel.** The constraint is $\rho^\top\!D\rho \propto I$, not $\rho^\top\rho = I$.
> Both contain the truth (each block orthogonal $\Rightarrow$ $\rho^\top\!W\rho \propto I$ for *any*
> positive diagonal $W$), so both are valid relaxations — but they are not equally good, see below.

**It all reduces to scalar graph theory.** On exact data, with $G = \operatorname{blkdiag}(R_p^\top)$
orthogonal and $A$ the weighted adjacency matrix $A_{pq} = c_{pq}$,

$$B \;=\; G\,(A \otimes I_3)\,G^\top,
\qquad\text{hence}\qquad
B_n \;=\; G\,(A_n \otimes I_3)\,G^\top,
\quad A_n = D^{-1/2} A D^{-1/2} .\tag{12a}$$

An orthogonal change of basis preserves eigenvalues and multiplicities, so **the spectrum of $B_n$ is
the spectrum of the scalar normalized adjacency $A_n$, with every eigenvalue tripled.**

**Why the weighting must be $D$.** Every positive diagonal $W$ gives a valid relaxation, but only
$W = D$ is *tight*. The true optimum of $(10)$ is $3\sum_p d_p$ — each edge term saturates at
$3c_{pq}$, since the trace of a rotation is at most 3 — while the relaxation with weighting $W$
returns $3\bigl(\sum_p w_p\bigr)\rho\bigl(W^{-1/2}AW^{-1/2}\bigr)$. Measured, with
$\sum_p d_p = 6$ so the true optimum is 18 in every column:

| $W$ | star | path $P_4$ | $K_3$ (regular) |
|---|---|---|---|
| $D$ | **18.00** (tight) | **18.00** (tight) | 18.00 |
| $I$ (plain Stiefel) | 20.78 | 19.42 | 18.00 |
| $D^2$ | 20.78 | 19.21 | 18.00 |
| random positive | 27.20 | 22.38 | 20.30 |

$D$ is tight for *every* graph because $\rho(A_n) = 1$ identically — the normalized adjacency always
has Perron root 1, with eigenvector $D^{1/2}\mathbf 1$. No other weighting has a graph-independent
Perron root, so none can be tight in general. The bound is invariant under $W \mapsto \alpha W$, so
$D$ is the unique tight direction up to scale. On a **regular** graph $D \propto I$ and the
distinction collapses — which is why the last column is all 18.

**The eigenvalue is 1, with multiplicity exactly 3.** Substituting the truth into $(9)$,

$$(B\rho)_q
= \sum_p c_{pq} \bigl(R_q^\top R_p\bigr) R_p^\top
= \Bigl( \sum_p c_{pq} \Bigr) R_q^\top
= d_q\, \rho_q
\qquad\Longrightarrow\qquad B\rho = D\rho, \quad B_n z = z .\tag{13}$$

$B_n$ has spectral radius $\le 1$ (each block row of $D^{-1}B$ is a convex combination of rotations,
so has operator norm at most 1), so the truth attains the **largest** eigenvalue. The multiplicity is
exactly 3, for two separate reasons:

- **At least 3.** $\rho$ is a $3n \times 3$ *matrix*: $(13)$ holds for each of its three columns
  separately, and $\rho^\top\rho = nI$ makes those columns mutually orthogonal. Those three
  directions are the three world axes — equivalently the 3 degrees of freedom of the global rotation
  $Q$ that the cost cannot see. **The 3 is the gauge.**
- **At most 3.** By $(12a)$ the question reduces to $A_n$, whose top eigenvalue 1 is *simple* on a
  connected component with $c_{pq} > 0$ (Perron–Frobenius). Read as a diffusion, eigenvalue-1 vectors
  are the stationary patterns, and a connected graph has exactly one; a second would have to change
  sign somewhere, and diffusion across the separating edge always mixes it.

So $\dim = 3 \times (\text{number of connected components})$ — the exact analogue of
$\dim\ker L = \text{number of components}$ in §5.3, with a full rotation per node in place of one
additive constant. (Verified: 3, 6, 9 eigenvalues equal to 1 for one, two and three disjoint
triangles.)

Undoing the normalization,

$$Y \;=\; D^{-1/2} Z_{1:3}, \qquad\text{blocks}\qquad Y_p \;=\; \nu_p\, R_p^\top Q \tag{14}$$

for an unknown common $Q \in \mathrm{O}(3)$ (the gauge) and positive per-node scalars $\nu_p$. On
exact data $\nu_p$ is the **same for every node**: the Perron eigenvector of $A_n$ is
$D^{1/2}\mathbf 1$, so $z_p = \sqrt{d_p}\,R_p^\top/\lVert\cdot\rVert$ and the $\sqrt{d_p}$ cancels
against $D^{-1/2}$ exactly. That uniform block scale is a second dividend of the $D$ weighting — it
is what makes the per-block rounding of $(15)$ equally well-posed at every node, however lopsided the
degrees.

### 4.3.1 Exact recovery

**On exact data, a connected component with $c_{pq} > 0$ recovers the ground truth exactly (up to the
gauge), in exact arithmetic.** The chain is:

1. The maximizing subspace is *unique*, because eigenvalue 1 has multiplicity exactly 3 and
   $\lambda_4 < 1$ strictly.
2. Any orthonormal basis of it is $Z = G(v \otimes I_3)Q/\lVert v\rVert$ for some $Q \in \mathrm{O}(3)$
   — this is where the eigensolver's orthonormality is load-bearing (§4.4).
3. $(14)$ gives $Y_p = \nu R_p^\top Q$ with $\nu$ uniform and positive.
4. Rounding is exact: if $\det Q = +1$ then $R_p^\top Q \in \mathrm{SO}(3)$ and $(15)$ returns it
   unchanged; if $\det Q = -1$ then *every* block has negative determinant, so the majority vote in
   §4.4 is unanimous and the column flip repairs it.
5. Anchoring cancels $Q$: $\hat R_a^\top \hat R_p = R_a^\top Q Q^\top R_p = R_a^\top R_p$.

Note step 1 is what upgrades tightness to *correctness*: a tight objective value alone would not pin
the maximizer if other maximizers existed.

**The spectral gap is not needed for this.** Verified numerically: a 300-node chain, whose gap has
collapsed to $5.5\times10^{-5}$, still recovers to $\sim\!10^{-8}$ rad; so do a star and a star with a
$10^{-6}$-weight edge. With zero noise there is nothing for a small gap to amplify.

### 4.3.2 Three diagnostics, measuring different things

All three come nearly free and are exposed on `RotationSyncResult` as `top_eigenvalues`,
`spectral_gap` and `algebraic_connectivity`.

| Quantity | Measures | Reads |
|---|---|---|
| $1 - \lambda_3$ | **inconsistency** of the measured relative rotations — do cycles close? | $0$ iff globally consistent; $\approx \varepsilon^2/3$ for typical per-edge error $\varepsilon$ |
| $\lambda_3 - \lambda_4$ | **identifiability** — is the top-3 eigenspace separated enough for the relaxation to round stably under noise? | small $\Rightarrow$ ill-conditioned, from either cause below |
| $\mu_2$ | **topology alone** — the normalized algebraic connectivity of the $n\times n$ scan graph, rotations not involved | the ceiling the gap cannot exceed |

**$\mu_2$ is the gap's pure-topology counterpart.** By $(12a)$ the two are *identical* on exact data,
$\lambda_3 - \lambda_4 = \mu_2$ (verified to machine precision on paths, $K_8$, stars and weak
bridges), and inconsistency only pulls the gap below it — $\mathrm{gap} \le \mu_2$ held in all of 400
randomized noisy graphs, with maximum ratio exactly $1.000000$. So comparing them **separates the two
causes of a small gap**:

- $\mathrm{gap} \approx \mu_2$, both small $\Rightarrow$ the **graph** is the limit. A path scales as
  $\mu_2 = O(1/n^2)$, so a long thin chain or a weak bridge is the usual culprit. Re-solving cannot
  help; add overlap across the bottleneck, raise a trusted bridge edge's $c_{pq}$, or attach an
  absolute prior. $\mu_2$ is computable from the $n\times n$ graph *before* any solve, so it can be
  evaluated and even optimized in advance.
- $\mathrm{gap} \ll \mu_2$ $\Rightarrow$ the **data** is the limit. The topology could support a
  well-determined answer; inconsistent relative rotations are dragging $\lambda_3$ down. Adding edges
  will not help.

$\mu_2$ also covers a blind spot of $1-\lambda_3$: on a tree or chain there are no cycles, so *any*
set of relative rotations is consistent and $1-\lambda_3 \equiv 0$ however noisy the data — a
40-node chain at $0.03$ rad noise reports $1-\lambda_3 \approx 10^{-16}$ while erring $12.6^\circ$.
For chain-like graphs $\mu_2$ and the gap are the only informative signals.

> **Numerical caveat.** $B_n$'s eigenvalues come in *triples*, and on a thin graph they are also
> tightly clustered — the hardest case for Lanczos. Requesting too few of them makes the iterative
> solver mis-resolve $\lambda_4$ badly in either direction (measured on a 120-node chain: $5\times$
> too large, or $\approx 0$). $\mu_2$ is immune, because the $n\times n$ spectrum is *not* degenerate,
> so it doubles as a validity bound: a reported gap above $\mu_2$, or a $\lambda_4$ at $1$ (impossible
> when multiplicity is exactly 3), means the solve failed and must be redone dense.

They are independent, so checking only the gap misses a whole failure mode. Measured on this code:

| Case | $\lambda_{1,2,3}$ | $1-\lambda_3$ | $\lambda_4$ |
|---|---|---|---|
| noise-free triangle | $1,1,1$ | $3\times10^{-16}$ | $-0.5$ |
| noisy 4-cycle, $\sigma = 0.05$ | $1, 0.99999, 0.99999$ | $9\times10^{-6}$ | $0.004$ |
| one poisoned edge, early GNC | $1, 0.9955, 0.9955$ | $4.5\times10^{-3}$ | $\approx 0$ |
| same, at GNC convergence | $1,1,1$ | $\approx 0$ | $5\times10^{-9}$ |

Rows 2 and 3 are the point: the noisy cycle is *consistent but tightly conditioned*, the poisoned
graph is *well conditioned but inconsistent*. Row 4 shows $1-\lambda_3$ falling as GNC rejects the
bad edge, so its trajectory over iterations is itself a useful signal. Note $\lambda_1 = 1$ whenever
the component is connected — use $\lambda_3$, or $\sum_{i\le3}(1-\lambda_i)$, never $\lambda_1$.

A *nearly* disconnected graph is one that almost has a fourth stationary pattern: two triangles
joined by a weight-$10^{-4}$ bridge keep multiplicity 3 but push $\lambda_4$ to $0.999967$. That is
the mechanism behind the weak-connectivity warning — connectivity buys separation, not correctness.
### 4.4 Rounding and reflection

Each block $Y_p$ is a scaled, noisy rotation. Project it back with the nearest-rotation map — a
*different* problem from $(8)$, namely
$\operatorname*{arg\,min}_{R \in \mathrm{SO}(3)} \lVert R - Y_p \rVert_F
= \operatorname*{arg\,max} \operatorname{tr}(R^\top Y_p)$:

$$Y_p = U \Sigma V^\top
\quad\Longrightarrow\quad
\tilde R_p = U \operatorname{diag}\!\bigl(1,\,1,\,\det(U V^\top)\bigr) V^\top,
\qquad R_p = \tilde R_p^\top\tag{15}$$

Note $U V^\top$ here, the transpose of the orientation in $(8)$; the final transpose is because the
blocks of $Y$ carry $R_p^\top$.

> **Orthonormality of $Z$ is load-bearing, not cosmetic.** Eigenvectors for *distinct* eigenvalues are
> orthogonal automatically ($B_n$ is symmetric), but eigenvalue 1 is 3-fold degenerate and *any* three
> independent vectors in that eigenspace are eigenvectors — the math alone does not orthogonalize
> them. The solvers do (`SelfAdjointEigenSolver` returns an orthogonal matrix; Spectra returns an
> orthonormal Ritz basis; measured $\lVert Z^\top Z - I\rVert_F \approx 10^{-16}$). This is exactly
> what makes $Q$ in $(14)$ lie in $\mathrm{O}(3)$ rather than $\mathrm{GL}(3)$: with a merely
> independent basis, $R_p^\top Q$ would be *skewed*, its polar factor would not equal it, and the skew
> would interact differently with each $R_p$ — so it would not even be a common gauge that anchoring
> could remove. The failure mode would be plausible-looking but wrong rotations, not a crash.
>
> Conversely, the arbitrariness the solver *does* have inside the eigenspace is harmless: two
> orthonormal bases of the same eigenspace differ by exactly a $3\times3$ orthogonal matrix, which is
> the gauge $Q$. The 3-fold degeneracy and the 3 DOF of a global rotation are the same fact.

The gauge $Q$ in $(14)$ lies in $\mathrm{O}(3)$, not $\mathrm{SO}(3)$: it may be a reflection, in
which case **every** block has negative determinant. Detect it by the sign of
$\sum_p \operatorname{sign}\det Y_p$ and, if negative, flip one column of $Y$ — which flips all
blocks simultaneously. A node whose determinant sign then still disagrees with its component is
genuinely inconsistent with the rest of the graph, and is flagged.

### 4.5 Gauge

The rounded rotations are determined up to the left action $R_p \leftarrow Q R_p$. Pinning the
anchor $a$,

$$R_p \;\leftarrow\; R_a^\top R_p \qquad (\text{so } R_a = I \text{ exactly})\tag{16}$$

This is cosmetic for the cost but makes the output deterministic across runs rather than drifting
with whatever basis the eigensolver returned.

---

## 5. Translation

With the rotations fixed, the weighted least-squares step of $(1)$ in the translations is **linear**:

$$
\operatorname*{arg\,min}_{\{t_i\}} \;
\sum_{(p,q) \in E} \; \sum_j \; w_j
\bigl\lVert (R_p a_j + t_p) - (R_q b_j + t_q) \bigr\rVert^2
\tag{17}
$$

### 5.1 The per-edge reduction is exact

Put $u = t_p - t_q$ and $c_j = R_q b_j - R_p a_j$, so the residual is $u - c_j$. With
$W_{pq} = \sum_j w_j$ and the weighted mean $d_{pq} = \bigl(\sum_j w_j c_j\bigr) / W_{pq}$,

$$
\sum_j w_j \lVert u - c_j \rVert^2
= W_{pq}\lVert u \rVert^2 - 2 W_{pq}\, u^\top d_{pq} + \sum_j w_j \lVert c_j \rVert^2
= W_{pq} \bigl\lVert u - d_{pq} \bigr\rVert^2
+ \underbrace{\Bigl( \sum_j w_j \lVert c_j\rVert^2 - W_{pq}\lVert d_{pq}\rVert^2 \Bigr)}_{\text{independent of } u}
\tag{18}
$$

So collapsing an edge's $m$ correspondences to the single target

$$d_{pq} \;=\; R_q\, \bar b_{pq} \;-\; R_p\, \bar a_{pq}
\qquad (\bar a, \bar b \text{ the weighted centroids of that edge's points})\tag{19}$$

loses nothing — it is an exact reformulation, not an approximation. Note $d_{qp} = -d_{pq}$.

It does discard the per-correspondence residuals, so a GNC loop must retain the raw points to
rebuild $r_j = (t_p - t_q) - c_j$ for the weight update $(3)$.

### 5.2 Normal equations are a graph Laplacian

The reduced problem is
$\min \sum_{(p,q)} W_{pq} \lVert t_p - t_q - d_{pq} \rVert^2$. Setting the gradient at node $p$ to
zero,

$$\sum_{q \sim p} W_{pq}\,(t_p - t_q) \;=\; \sum_{q \sim p} W_{pq}\, d_{pq} .\tag{20}$$

The left side is exactly the **weighted graph Laplacian** applied to the translations, so stacking
the nodes as rows of $T \in \mathbb{R}^{n \times 3}$,

$$L\,T = \mathcal{B},
\qquad
L_{pp} = \sum_{q\sim p} W_{pq},
\quad
L_{pq} = -W_{pq},
\quad
\mathcal{B}_{[p]} = \sum_{q \sim p} W_{pq}\, d_{pq}^\top .\tag{21}$$

Three right-hand sides — $x$, $y$, $z$ — share one matrix, so one factorization solves all three.

### 5.3 Singularity is the gauge, and consistency is automatic

$L \mathbf{1} = 0$: the constant shift is precisely the translation gauge of §1, so $L$ is singular
with a one-dimensional nullspace per connected component. This is the scalar case of the same
statement §4.3 makes for rotation: the connection Laplacian $D - B$ of $(10a)$ annihilates the true
$\rho$, with a null space of dimension $3$ per component instead of $1$ — a whole rotation per node
in place of one additive constant. The system is nonetheless consistent,
because the antisymmetry $d_{qp} = -d_{pq}$ gives

$$\mathbf{1}^\top \mathcal{B}
= \sum_{(p,q) \in E} \bigl( W_{pq} d_{pq}^\top - W_{pq} d_{pq}^\top \bigr)
= 0\tag{22}$$

so $\mathcal{B}$ lies in the range of $L$. Fixing the gauge by pinning the anchor $t_a = 0$ and
deleting row and column $a$ yields the **grounded Laplacian**, which is positive definite for a
connected component and stays sparse — so a sparse Cholesky factorization applies directly.

> The textbook alternative, factorizing $L + \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$ to impose
> $\sum_p t_p = 0$, is both denser (that rank-one term is full) and pointless here, since anchoring
> discards its zero-mean gauge anyway.

### 5.4 Uncertainty

Because $(21)$ is linear, uncertainty propagates in closed form. The **effective resistance**
between two nodes,

$$R_{\mathrm{eff}}(p,q) \;=\; (e_p - e_q)^\top L^{+} (e_p - e_q),\tag{23}$$

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
| Scale **all** $W_{pq}$ by $\alpha > 0$ | none on translation | scales $L$ and $\mathcal{B}$ equally in $(21)$ |
| Uniform noise bounds $\delta_i \equiv \delta$ | none on either weighting | a common factor on all weights |
| Recenter cloud $p$ by $o_p$ | none on translation | shifts $d_{pq}$ by $R_p o_p - R_q o_q$, which the undo $t_p \leftarrow t_p - R_p o_p$ cancels exactly |
| Magnitude in $\rho^\top\! D\rho \propto I$ | none | objective is homogeneous of degree 2, §4.3 |
| Weighting $W \mapsto \alpha W$ in the constraint | none | $\sum_p w_p$ and $\rho(W^{-1/2}AW^{-1/2})$ scale inversely, §4.3 |
| Dropping the $D^{-1/2}$ (use raw $B$) | none **on exact data** | top-3 blocks become $(v_1)_p R_p^\top$ with $v_1>0$ Perron; rounding is scale-invariant, so the same rotations come out. It is *not* harmless in general: the relaxation is then loose, block norms are degree-dependent (worse conditioning at weakly-connected nodes), and $\lambda=1$ is lost so §4.3.2 breaks |

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

**Refinement.** The two stages above are separate relaxations, so a final joint Levenberg–Marquardt
pass polishes $(R, t)$ together on the raw-point cost of $(1)$. It commits only cost-decreasing
steps, so it cannot return something worse than the spectral estimate; being local, it will not
repair a pose that landed in the wrong basin. Gated by `multiview_refine_iterations` ($0$ = off).

**Finally**, re-gauge each component to its anchor (§4.5, §5.3).

Where each piece lives:

| Section | Code |
|---|---|
| §2 GNC weight update, annealing | `multiview.cc` (`gncUpdateWeights`, `gncStep`) |
| §4.1 per-edge reduction | `rotation_sync.cc`, via `utils::svdRot` |
| §4.2–4.5 spectral solve | `rotation_sync.cc` (`synchronizeRotations`) |
| §4.3.2 spectrum diagnostics | `RotationSyncResult::spectral_gap`, `::top_eigenvalues` |
| §5 Laplacian solve, diagnostics | `translation_sync.cc` (`synchronizeTranslations`) |
| §6 bound composition | `timResidualNoiseBound`, `pointResidualNoiseBound` |
| §8 refinement | `pose_refine.cc` (`refinePoses`) |
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
- A. S. Bandeira, A. Singer, D. A. Spielman, *A Cheeger Inequality for the Graph Connection
  Laplacian* — the $D - B$ operator of $(10a)$, its null space, and the link between its spectral gap
  and graph connectivity.
- D. M. Rosen, L. Carlone, A. S. Bandeira, J. J. Leonard, *SE-Sync: A Certifiably Correct Algorithm
  for Synchronization over the Special Euclidean Group* — the semidefinite alternative to §4.3's
  spectral relaxation, with an a posteriori global-optimality certificate.
- F. Arrigoni, A. Fusiello, *Synchronization Problems in Computer Vision with Closed-Form
  Solutions* — survey of rotation and translation synchronization.
