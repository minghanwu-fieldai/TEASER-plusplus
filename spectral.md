## Rotation

    argmin_{R_i in SO(3)}  sum_{(p,q) in E} sum_j w_j || R_p x_j - R_q y_j ||^2

Implemented in `teaser/src/rotation_sync.cc` (`synchronizeRotations`).

```python
# ---------- Step 0: per-edge reduction ----------
for (p, q) in E:
    X = p_tims[p,q]                      # (m,3) TIM vectors in p's frame -- NOT raw points
    Y = q_tims[p,q]                      # (m,3) TIM vectors in q's frame
    w = weights[p,q]                     # (m,) GNC line-process weights, change every iteration

    if w.sum() < eps: continue           # GNC rejected every correspondence -> edge constrains nothing
    M[p,q] = X.T @ (w[:,None] * Y)       # 3x3 cross-covariance; TIMs are translation-free, so no centering
    if norm(M[p,q]) < eps: continue      # degenerate: the SVD would return an arbitrary rotation

    U, sig, Vt = svd(M[p,q]);  V = Vt.T
    Rhat[p,q] = V @ diag(1,1,det(V @ U.T)) @ U.T   # argmax_A trace(M A) == R_q^T R_p -- V U^T, NOT U V^T
    sigma[p,q] = 2*delta[p] + 2*delta[q] # TIM = difference of 2 points, and both scans are noisy
    c[p,q] = confidence[p,q] / sigma[p,q]**2       # the ONLY channel a per-scan noise bound can act through

# Rescaling X, Y by 1/sigma would be a no-op: it scales M, and Rhat is M's polar factor.
# Uniform c is likewise a no-op: a common factor cancels out of Bn.

# ---------- Step 1: components over the SURVIVING edges ----------
for comp in connected_components(V, E_kept):
    n = len(comp);  dim = 3*n             # nodes with no surviving edge: R = I (or previous), valid = False
    if n < 2: R[comp[0]] = I; continue

    # ---------- Step 2: assemble the normalized matrix ----------
    d[p] = sum(c[p,q] for q in nbrs(p))   # weighted degree
    Bn = sparse((dim, dim))               # 18 nonzeros per edge
    for (p,q) in edges(comp):
        s = c[p,q] / sqrt(d[p]*d[q])      # D^-1/2 folded into the entries, never materialized
        Bn[3q:3q+3, 3p:3p+3] += s * Rhat[p,q]      # += so parallel edges accumulate
        Bn[3p:3p+3, 3q:3q+3] += s * Rhat[p,q].T    # symmetric; spectral radius <= 1

    # ---------- Step 3: top-3 eigenspace ----------
    if dim <= 60:
        lams, Z = eigh(dense(Bn))         # ASCENDING -> top-3 is the last 3 columns
        gap = lams[dim-3] - lams[dim-4]
    else:
        lams, Z = eigsh(Bn, k=min(6,dim-1), which='LA')   # DESCENDING; fall back to dense if it fails
        gap = lams[2] - lams[3]
    Y = D_ihalf @ Z[:, top3]              # undo the normalization -> blocks ~ v(p) R_p^T Q
    if gap < tau_gap: warn(...)           # DIAGNOSTIC: weak connectivity, poorly determined

    # ---------- Step 4: fix the global reflection, then round ----------
    dets = [ det(Y[3p:3p+3, :]) for p in comp ]
    if sum(sign(dets)) < 0:               # gauge Q had det -1; SAME sign for all p
        Y[:, 2] *= -1;  dets = -dets      # one column flip flips every block at once

    for p in comp:
        U, _, Vt = svd(Y[3p:3p+3, :])
        Rp = U @ diag(1,1,det(U @ Vt)) @ Vt        # nearest rotation to Y_p -- U V^T here, unlike Rhat
        R[p] = Rp.T                       # Y blocks carry R_p^T
        reliable[p] = dets[p] > 0         # per-node disagreement => bad node

    # ---------- Step 5: fix the gauge ----------
    a = min(comp)                         # cost is invariant under R_p -> Q R_p
    for p in comp: R[p] = R[a].T @ R[p]   # anchor becomes exactly I; keeps output deterministic
```

## Translation

    argmin_{t_i}  sum_{(p,q) in E} sum_j w_j || (R_p p_j + t_p) - (R_q q_j + t_q) ||^2

Normal equations of a weighted graph Laplacian, `L T = Bm`, three RHS. Plain LS, whereas
multiview.md specifies component-wise TLS -- correct as a GNC inner step, not as a one-shot.
Not implemented yet.

```python
# ---------- Step 0: per-edge reduction ----------
# No recentering: shifting cloud p by o[p] moves d[p,q] by (R_p o_p - R_q o_q), and the undo
# t_p -= R_p o_p cancels it exactly. Also ill-defined, since the weights are per-EDGE.
for (p,q) in E_kept:
    w = weights[p,q]
    P = p_pts[p,q];  Q = q_pts[p,q]       # (m,3) RAW points -- differencing into TIMs would kill t
    sigma[p,q] = delta[p] + delta[q]      # raw points: NO factor 2, unlike the rotation stage
    W[p,q]    = w.sum() / sigma[p,q]**2   # edge information
    pbar[p,q] = (w @ P) / w.sum()         # weighted centroid of p's points on this edge
    qbar[p,q] = (w @ Q) / w.sum()
    d[p,q]    = R[q] @ qbar[p,q] - R[p] @ pbar[p,q]   # target for t_p - t_q

# The reduction is EXACT: with c_j = R_q q_j - R_p p_j,
#   sum_j w_j ||(t_p - t_q) - c_j||^2 == w.sum() * ||(t_p - t_q) - d[p,q]||^2 + const
# It discards the per-correspondence residuals, so a GNC loop must keep P, Q to rebuild r_j.

# ---------- Step 1: assemble the Laplacian, solve 3 RHS at once ----------
for comp in connected_components(V, E_kept):
    n = len(comp)
    if n == 1: t[comp[0]] = 0; continue

    L = sparse((n,n));  Bm = zeros((n,3))
    for (p,q) in edges(comp):
        Wq = W[p,q]
        L[p,p] += Wq;  L[q,q] += Wq;  L[p,q] -= Wq;  L[q,p] -= Wq
        Bm[p]  += Wq * d[p,q]
        Bm[q]  -= Wq * d[p,q]             # d[q,p] == -d[p,q]; keeps 1^T Bm == 0

    a   = min(comp)                       # gauge: pin the anchor, the SAME node the rotation stage pinned
    idx = [p for p in comp if p != a]     # NOT L + ones(n,n)/n: dense, and the anchoring discards it
    F   = cholesky(L[idx][:,idx])         # grounded Laplacian: sparse, and PD when comp is connected
    T   = F.solve(Bm[idx])                # (n-1) x 3, ONE factorization
    t[a] = 0;  t[idx] = T

    # ---------- Step 2: diagnostics ----------
    sigma_res = sqrt(weighted_residual_sum / max(1, 3*(len(edges(comp)) - (n-1))))
    for (p,q) in pairs_of_interest:
        v = (e_p - e_q)[idx]              # anchor entry drops out; t[a] == 0
        R_eff = v @ F.solve(v)            # effective resistance; never form pinv(L), which is dense
        report relative-translation sigma = sigma_res * sqrt(R_eff)
    report lambda_2(L)                    # Fiedler value; weak => add edges, don't refine more
```
