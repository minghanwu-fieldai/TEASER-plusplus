/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/SparseCore>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include "rotation_sync.h"
#include "sync_components.h"
#include "teaser/utils.h"

namespace {

// Below this the cross-covariance carries no orientation information and its SVD would return an
// arbitrary rotation, so the edge is dropped instead.
constexpr double kMinCrossCovNorm = 1e-12;

/**
 * Nearest rotation to A: argmin_{R in SO(3)} ||R - A||_F, equivalently argmax trace(R^T A).
 *
 * Note this is NOT the same problem as teaser::utils::svdRot, which maximizes trace(R H); the two
 * differ by a transpose, hence U * V^T here versus V * U^T there. The determinant guard is the same
 * idiom: flipping the last column of V is the cheapest way to land on the proper-rotation branch,
 * because Eigen orders singular values descending so the sacrificed axis is the least-supported one.
 */
Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d& A) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d& V = svd.matrixV();
  if (U.determinant() * V.determinant() < 0) {
    U.col(2) *= -1;
  }
  return U * V.transpose();
}

/**
 * One edge of the AUGMENTED per-component graph, in component-local indices.
 *
 * Real scan-to-scan edges carry M = Rhat (a rotation). The optional virtual "world" node carries
 * M = g_i * up^T, which is rank one -- deliberately not polar-projected, since a rank-one matrix has
 * no meaningful rotation factor. Both go through the same assembly, which is what lets the
 * synchronization matrix and the mu_2 diagnostic be built from one list.
 */
struct AugEdge {
  int lp;               // local index of p (0 is the virtual node when present)
  int lq;               // local index of q
  double c;             // edge confidence
  Eigen::Matrix3d M;    // block placed at [lq][lp]; its transpose goes at [lp][lq]

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Smallest rotation taking unit vector `a` to unit vector `b`.
 *
 * Used for the upright gauge instead of the Procrustes/SVD solution. Both align the summed gravity
 * with world up, but the Procrustes matrix sum_i up * (R_i g_i)^T is ALWAYS rank one (up is a fixed
 * vector), so its SVD leaves the yaw about `up` to an arbitrary null-space basis -- the answer would
 * not be reproducible. Gravity genuinely does not determine yaw, so pick the canonical minimal
 * rotation and leave it at that.
 */
Eigen::Matrix3d minimalRotation(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const Eigen::Vector3d v = a.cross(b);
  const double c = a.dot(b);
  const double s = v.norm();
  if (s < 1e-12) {
    if (c > 0) {
      return Eigen::Matrix3d::Identity();
    }
    // Antipodal: rotate by pi about any axis perpendicular to a. Pick one deterministically so the
    // gauge stays reproducible.
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
    if (std::abs(a.x()) > 0.9) {
      axis = Eigen::Vector3d::UnitY();
    }
    axis = (axis - a.dot(axis) * a).normalized();
    return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
  }
  return Eigen::AngleAxisd(std::atan2(s, c), v / s).toRotationMatrix();
}

/** An edge that survived preprocessing, reduced to just what synchronization needs. */
struct ReducedEdge {
  int p;
  int q;
  double c;
  Eigen::Matrix3d Rhat; // approximately R_q^T R_p

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/** NaN-filled placeholder for eigenvalues that were not computed. Eigenvalues can legitimately be
 * negative, so a negative sentinel would be ambiguous. */
Eigen::Vector4d unavailableEigenvalues() {
  return Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
}

/**
 * Top-3 eigenspace of the symmetric matrix Bn (dimension dim), plus its four largest eigenvalues in
 * DESCENDING order (trailing entries NaN when fewer than four are available). The spectral gap is
 * derived from these by the caller, so there is one source of truth for the spectrum.
 *
 * The three returned columns are an arbitrary basis of that eigenspace; their order does not matter
 * because any change of basis is absorbed into the global gauge.
 */
bool topThreeEigenspaceDense(const Eigen::SparseMatrix<double>& Bn, int dim, Eigen::MatrixXd* Z,
                             Eigen::Vector4d* top4) {
  const Eigen::MatrixXd dense = Eigen::MatrixXd(Bn);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(dense);
  if (eig.info() != Eigen::Success) {
    return false;
  }
  // Eigen sorts eigenvalues ascending, so the top-3 eigenspace is the last three columns.
  *Z = eig.eigenvectors().rightCols(3);
  const Eigen::VectorXd& asc = eig.eigenvalues();
  *top4 = unavailableEigenvalues();
  for (int k = 0; k < 4 && k < dim; ++k) {
    (*top4)(k) = asc(dim - 1 - k); // ascending -> descending
  }
  return true;
}

bool topThreeEigenspaceSparse(const Eigen::SparseMatrix<double>& Bn, int dim, Eigen::MatrixXd* Z,
                              Eigen::Vector4d* top4) {
  // B_n's eigenvalues come in TRIPLES (its spectrum is the scalar normalized adjacency's, tripled),
  // so asking for 6 requests exactly two complete triples and puts the Lanczos boundary inside a
  // degenerate cluster -- it then returns a spurious lambda_4, badly wrong in either direction.
  // Requesting several triples with a generous Krylov basis resolves it; measured on a 120-node
  // chain, nev=6/ncv=20 gave a gap 5x too large and nev=12/18 gave ~0, while nev=24/ncv=145 was
  // exact and no slower than the dense solve. The caller additionally validates the result.
  const int nev = std::min(24, dim - 1);
  if (nev < 3) {
    return false;
  }
  const int ncv = std::min(dim, std::max(6 * nev + 1, 150));
  // Spectra 0.x API, matching the usage in certification.cc: op is passed by pointer, and success
  // is reported as Spectra::SUCCESSFUL.
  Spectra::SparseSymMatProd<double> op(Bn);
  Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, Spectra::SparseSymMatProd<double>> eigs(
      &op, nev, ncv);
  eigs.init();
  const int nconv = eigs.compute();
  if (eigs.info() != Spectra::SUCCESSFUL || nconv < 3) {
    return false;
  }
  // For LARGEST_ALGE, Spectra returns eigenvalues in descending order. Only the converged ones are
  // trustworthy, so anything beyond nconv stays NaN.
  *Z = eigs.eigenvectors().leftCols(3);
  const Eigen::VectorXd vals = eigs.eigenvalues();
  *top4 = unavailableEigenvalues();
  for (int k = 0; k < 4 && k < nconv && k < vals.size(); ++k) {
    (*top4)(k) = vals(k);
  }
  return true;
}

/**
 * Normalized algebraic connectivity (Fiedler value) mu_2 of the n-by-n scan graph whose weights are
 * the edge confidences c_pq: the second smallest eigenvalue of I - D^-1/2 A D^-1/2, equivalently
 * 1 - alpha_2 with alpha_2 the second largest eigenvalue of the normalized adjacency.
 *
 * Pure topology -- the relative rotations never enter. On exact data this equals the spectral gap of
 * the 3n-by-3n problem identically, because B_n is orthogonally similar to A_n kron I_3; comparing
 * the two separates a thin graph from inconsistent data (see RotationSyncResult).
 *
 * `aug_edges` and `d` are the same edge list and degrees used to assemble B_n -- including the
 * virtual world node when the upright prior is enabled -- so the two diagnostics always describe the
 * graph that was actually solved.
 *
 * The n-by-n problem is 27x cheaper than the 3n-by-3n one, so dense stays affordable to the same
 * FLOP budget at n <= dense_max_dim; above that use the iterative solver.
 */
double normalizedAlgebraicConnectivity(const std::vector<AugEdge>& aug_edges,
                                       const std::vector<double>& d, int n, int dense_max_dim) {
  if (n < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::vector<double> d_inv_sqrt(n);
  for (int k = 0; k < n; ++k) {
    d_inv_sqrt[k] = d[k] > 0 ? 1.0 / std::sqrt(d[k]) : 0.0;
  }
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(aug_edges.size() * 2);
  for (const AugEdge& e : aug_edges) {
    const double v = e.c * d_inv_sqrt[e.lp] * d_inv_sqrt[e.lq];
    triplets.emplace_back(e.lp, e.lq, v);
    triplets.emplace_back(e.lq, e.lp, v);
  }
  Eigen::SparseMatrix<double> An(n, n);
  An.setFromTriplets(triplets.begin(), triplets.end()); // sums parallel edges, as for B_n

  // alpha_1 == 1 always (Perron, eigenvector D^1/2 * 1); we need alpha_2.
  if (n <= dense_max_dim) {
    const Eigen::MatrixXd An_dense = An; // named: eig(Eigen::MatrixXd(An)) is a function decl
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(An_dense);
    if (eig.info() != Eigen::Success) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 - eig.eigenvalues()(n - 2); // ascending: second largest
  }
  const int nev = std::min(4, n - 1);
  if (nev < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int ncv = std::min(n, std::max(2 * nev + 1, 20));
  Spectra::SparseSymMatProd<double> op(An);
  Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, Spectra::SparseSymMatProd<double>> eigs(
      &op, nev, ncv);
  eigs.init();
  const int nconv = eigs.compute();
  if (eigs.info() != Spectra::SUCCESSFUL || nconv < 2) {
    // Fall back to dense rather than reporting nothing: n-by-n is cheap even when 3n-by-3n was not.
    const Eigen::MatrixXd An_dense = An; // named: eig(Eigen::MatrixXd(An)) is a function decl
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(An_dense);
    if (eig.info() != Eigen::Success) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 - eig.eigenvalues()(n - 2);
  }
  return 1.0 - eigs.eigenvalues()(1); // descending: second largest
}

/**
 * Reject an iterative solve whose spectrum cannot be right. Both checks are provable, and both fire
 * in practice on clustered (thin-graph) spectra -- exactly the regime the gap diagnostic is for.
 *
 * 1. lambda_4 must sit strictly below 1. Eigenvalue 1 has multiplicity EXACTLY 3 on a connected
 *    component (Perron-Frobenius on the normalized adjacency), so a fourth eigenvalue at 1 means the
 *    solver failed to separate the leading cluster and is reporting duplicates.
 * 2. lambda_3 - lambda_4 <= mu_2. The gap equals mu_2 on exact data and inconsistency only pulls it
 *    down (verified over 400 randomized noisy graphs, max ratio 1.000000), so a gap above mu_2 is
 *    impossible and indicates a failed solve.
 */
bool sparseSpectrumPlausible(const Eigen::Vector4d& top4, double mu2) {
  if (std::isnan(top4(2)) || std::isnan(top4(3))) {
    return false;
  }
  if (top4(3) > 1.0 - 1e-9) {
    return false;
  }
  const double gap = top4(2) - top4(3);
  if (std::isfinite(mu2) && gap > mu2 * (1.0 + 1e-6) + 1e-12) {
    return false;
  }
  return true;
}

} // namespace

double teaser::timResidualNoiseBound(double delta_p, double delta_q) {
  return 2.0 * delta_p + 2.0 * delta_q;
}

teaser::RotationSyncResult
teaser::synchronizeRotations(int num_nodes, const std::vector<teaser::RotationSyncEdge>& edges,
                             const teaser::RotationSyncParams& params,
                             const std::vector<Eigen::Matrix3d>* initial,
                             const std::vector<double>* node_noise_bounds,
                             const std::vector<Eigen::Vector3d>* node_gravity) {
  teaser::RotationSyncResult result;
  if (num_nodes <= 0) {
    return result;
  }

  const bool use_initial = initial != nullptr && static_cast<int>(initial->size()) == num_nodes;
  const bool use_noise_bounds =
      node_noise_bounds != nullptr && static_cast<int>(node_noise_bounds->size()) == num_nodes;
  if (node_noise_bounds != nullptr && !use_noise_bounds) {
    std::cerr << "[teaser::rotation_sync] Warning: node_noise_bounds has "
              << node_noise_bounds->size() << " entries but there are " << num_nodes
              << " nodes; ignoring it.\n";
  }

  // Per-scan gravity. A zero vector marks a scan with no reading; such nodes are skipped rather
  // than treated as having a degenerate up direction.
  bool use_gravity =
      node_gravity != nullptr && static_cast<int>(node_gravity->size()) == num_nodes;
  if (node_gravity != nullptr && !use_gravity) {
    std::cerr << "[teaser::rotation_sync] Warning: node_gravity has " << node_gravity->size()
              << " entries but there are " << num_nodes << " nodes; ignoring it.\n";
  }
  Eigen::Vector3d world_up = Eigen::Vector3d::UnitZ();
  if (use_gravity) {
    if (params.world_up.norm() < 1e-12) {
      std::cerr << "[teaser::rotation_sync] Warning: params.world_up is degenerate; ignoring the "
                   "upright prior.\n";
      use_gravity = false;
    } else {
      world_up = params.world_up.normalized();
    }
  }
  // Normalized gravity per node, zero where unavailable.
  std::vector<Eigen::Vector3d> gravity(num_nodes, Eigen::Vector3d::Zero());
  if (use_gravity) {
    for (int i = 0; i < num_nodes; ++i) {
      const Eigen::Vector3d& g = (*node_gravity)[i];
      if (g.norm() > 1e-12) {
        gravity[i] = g.normalized();
      }
    }
  }
  const bool use_vnode = use_gravity && params.upright_prior_eta > 0;

  result.rotations.assign(num_nodes, Eigen::Matrix3d::Identity());
  if (use_initial) {
    result.rotations = *initial;
  }
  result.valid.assign(num_nodes, false);
  result.reliable.assign(num_nodes, false);
  result.component.assign(num_nodes, -1);
  result.tilt_error.assign(num_nodes, std::numeric_limits<double>::quiet_NaN());

  // ---------- Step 0: reduce each edge to its relative rotation ----------
  std::vector<ReducedEdge> kept;
  kept.reserve(edges.size());
  for (const auto& e : edges) {
    if (e.p < 0 || e.q < 0 || e.p >= num_nodes || e.q >= num_nodes || e.p == e.q) {
      continue;
    }
    const Eigen::Index m = e.p_tims.cols();
    if (m == 0 || e.q_tims.cols() != m || e.w.cols() != m) {
      continue;
    }
    if (!(e.confidence > 0)) {
      continue;
    }
    // Per-scan noise bounds. With a per-point bound per scan, every TIM correspondence on this edge
    // shares one sigma_pq, so the whole edge is simply weighted by its information 1 / sigma_pq^2.
    //
    // Rescaling the correspondences themselves by 1 / sigma_pq -- the obvious reading of "put the
    // residual in units of its own noise bound" -- would be a NO-OP here: it scales M by
    // 1 / sigma_pq^2, and Rhat is M's polar factor, which is invariant to any positive scaling of
    // M. The magnitude of M is discarded on purpose (that is what reducing each edge to a pure
    // relative rotation buys), so the edge weight is the only channel through which a per-scan
    // noise bound can reach the answer. Per-CORRESPONDENCE bounds would be different: those vary
    // within an edge, do not factor out of M, and would change Rhat too.
    double edge_confidence = e.confidence;
    if (use_noise_bounds) {
      const double sigma =
          teaser::timResidualNoiseBound((*node_noise_bounds)[e.p], (*node_noise_bounds)[e.q]);
      if (!(sigma > 0)) {
        continue; // both scans declared noiseless; the weight would be infinite
      }
      edge_confidence /= sigma * sigma;
    }
    // An edge whose correspondences have all been rejected by GNC constrains nothing.
    if (e.w.sum() < params.min_edge_mass) {
      continue;
    }
    // M[p,q] = sum_j w_j x_j y_j^T, the weighted cross-covariance of the two TIM sets. Used as
    // given: TIMs are already translation-free, so there is no centroid to subtract.
    const Eigen::Matrix3d M = e.p_tims * e.w.asDiagonal() * e.q_tims.transpose();
    if (M.norm() < kMinCrossCovNorm) {
      continue;
    }
    // argmax_{A in SO(3)} trace(M A) == R_q^T R_p. utils::svdRot builds exactly this M internally
    // (H = X W Y^T) and returns that maximizer with the determinant guard already applied, so the
    // whitening step is just a call to the same kernel the pairwise GNC rotation solver uses.
    ReducedEdge r;
    r.p = e.p;
    r.q = e.q;
    r.c = edge_confidence;
    r.Rhat = teaser::utils::svdRot(e.p_tims, e.q_tims, e.w);
    kept.push_back(r);
  }

  // ---------- Step 1: connected components over the surviving edges ----------
  std::vector<std::pair<int, int>> kept_pairs;
  kept_pairs.reserve(kept.size());
  for (const auto& r : kept) {
    kept_pairs.emplace_back(r.p, r.q);
  }
  const teaser::SyncComponents comps = teaser::labelSyncComponents(num_nodes, kept_pairs);
  const std::vector<std::vector<int>>& comp_nodes = comps.nodes;
  result.component = comps.component;
  result.num_components = comps.num_components;
  result.spectral_gap.assign(result.num_components, -1.0);
  result.top_eigenvalues.assign(result.num_components, unavailableEigenvalues());
  result.algebraic_connectivity.assign(result.num_components,
                                       std::numeric_limits<double>::quiet_NaN());
  result.gauge_upright.assign(result.num_components, false);
  if (result.num_components == 0) {
    return result;
  }
  if (result.num_components > 1) {
    std::cerr << "[teaser::rotation_sync] Warning: graph has " << result.num_components
              << " connected components; synchronizing each independently.\n";
  }

  // Bucket the edges by component and map global node ids to component-local ids.
  std::vector<int> local_id(num_nodes, -1);
  for (const auto& nodes : comp_nodes) {
    for (size_t k = 0; k < nodes.size(); ++k) {
      local_id[nodes[k]] = static_cast<int>(k);
    }
  }
  std::vector<std::vector<const ReducedEdge*>> comp_edges(result.num_components);
  for (const auto& r : kept) {
    comp_edges[result.component[r.p]].push_back(&r);
  }

  for (int comp = 0; comp < result.num_components; ++comp) {
    const std::vector<int>& nodes = comp_nodes[comp];
    const int n = static_cast<int>(nodes.size());

    // A component is built from edges, so it always holds at least two nodes. Guard anyway: a lone
    // node's rotation is pure gauge, so the identity is correct.
    if (n < 2) {
      result.rotations[nodes[0]] = Eigen::Matrix3d::Identity();
      result.valid[nodes[0]] = true;
      result.reliable[nodes[0]] = true;
      continue;
    }

    // Does this component have any gravity at all?
    int gravity_count = 0;
    for (int k = 0; k < n; ++k) {
      if (gravity[nodes[k]].squaredNorm() > 0) {
        ++gravity_count;
      }
    }
    // The virtual node is added PER COMPONENT. A single world node joined to every scan would
    // otherwise merge disconnected components, reporting them as one while their relative yaw and
    // translation remain unconstrained -- so components are labelled on the real edges only (above)
    // and each is augmented separately here.
    const bool vnode = use_vnode && gravity_count > 0;
    const int voff = vnode ? 1 : 0; // real node k lives at local index voff + k; 0 is the world
    const int n_aug = n + voff;
    const int dim = 3 * n_aug;

    // ---------- Step 2: assemble the normalized synchronization matrix ----------
    std::vector<AugEdge> aug_edges;
    aug_edges.reserve(comp_edges[comp].size() + (vnode ? n : 0));
    for (const ReducedEdge* r : comp_edges[comp]) {
      aug_edges.push_back({local_id[r->p] + voff, local_id[r->q] + voff, r->c, r->Rhat});
    }
    if (vnode) {
      // Gravity as an ordinary edge to the world node: the reward eta * up^T R_i g_i equals
      // 2 * eta * tr(rho_i^T (g_i up^T)^T rho_0) at rho_0 = I, so the block at [i][0] is
      // eta * g_i up^T. Rank one on purpose -- it encodes only the two DOF gravity actually fixes,
      // and must NOT be polar-projected.
      for (int k = 0; k < n; ++k) {
        const Eigen::Vector3d& g = gravity[nodes[k]];
        if (g.squaredNorm() == 0) {
          continue;
        }
        aug_edges.push_back({0, voff + k, params.upright_prior_eta, g * world_up.transpose()});
      }
    }

    std::vector<double> d(n_aug, 0.0);
    for (const AugEdge& e : aug_edges) {
      d[e.lp] += e.c;
      d[e.lq] += e.c;
    }
    std::vector<double> d_inv_sqrt(n_aug);
    for (int k = 0; k < n_aug; ++k) {
      d_inv_sqrt[k] = d[k] > 0 ? 1.0 / std::sqrt(d[k]) : 0.0;
    }

    // Bn = D^{-1/2} B D^{-1/2}, with block (q,p) = c * M and block (p,q) = c * M^T. The D^{-1/2}
    // scaling is folded straight into the triplets rather than materialized.
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(aug_edges.size() * 18);
    for (const AugEdge& e : aug_edges) {
      const double s = e.c * d_inv_sqrt[e.lp] * d_inv_sqrt[e.lq];
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const double v = s * e.M(a, b);
          triplets.emplace_back(3 * e.lq + a, 3 * e.lp + b, v);
          triplets.emplace_back(3 * e.lp + b, 3 * e.lq + a, v);
        }
      }
    }
    Eigen::SparseMatrix<double> Bn(dim, dim);
    // setFromTriplets sums duplicates, so parallel edges on the same node pair accumulate rather
    // than one of them silently winning.
    Bn.setFromTriplets(triplets.begin(), triplets.end());

    // ---------- Step 3: top-3 eigenspace ----------
    // mu_2 first: it is cheap (n-by-n, and its spectrum is NOT degenerate so it is reliable even
    // when B_n's is clustered) and it doubles as a validity bound on the iterative solve below.
    const double mu2 = normalizedAlgebraicConnectivity(aug_edges, d, n_aug, params.dense_max_dim);
    result.algebraic_connectivity[comp] = mu2;

    Eigen::MatrixXd Z;
    Eigen::Vector4d top4 = unavailableEigenvalues();
    bool ok = false;
    if (dim > params.dense_max_dim) {
      ok = topThreeEigenspaceSparse(Bn, dim, &Z, &top4);
      if (ok && !sparseSpectrumPlausible(top4, mu2)) {
        std::cerr << "[teaser::rotation_sync] Warning: iterative eigensolver returned an "
                     "inconsistent spectrum on component "
                  << comp << " (lambda_3=" << top4(2) << ", lambda_4=" << top4(3)
                  << ", mu_2=" << mu2 << "); redoing it dense.\n";
        ok = false;
      } else if (!ok) {
        std::cerr << "[teaser::rotation_sync] Warning: iterative eigensolver failed on component "
                  << comp << "; falling back to a dense decomposition.\n";
      }
    }
    if (!ok) {
      ok = topThreeEigenspaceDense(Bn, dim, &Z, &top4);
    }
    if (!ok) {
      std::cerr << "[teaser::rotation_sync] Warning: eigendecomposition failed on component "
                << comp << "; leaving its nodes unestimated.\n";
      continue;
    }
    result.top_eigenvalues[comp] = top4;
    // Derived from the spectrum, so the two diagnostics cannot disagree. NaN lambda_4 (fewer than
    // four eigenvalues available) keeps the documented -1 "not computable" marker for the gap.
    const double gap = std::isnan(top4(2)) || std::isnan(top4(3)) ? -1.0 : top4(2) - top4(3);
    result.spectral_gap[comp] = gap;
    if (gap >= 0 && gap < params.gap_threshold) {
      std::cerr << "[teaser::rotation_sync] Warning: component " << comp << " has spectral gap "
                << gap << " (< " << params.gap_threshold << "); rotations are poorly determined."
                << " Normalized algebraic connectivity mu_2 = " << mu2
                << (std::isfinite(mu2) && gap < 0.5 * mu2
                        ? " -- gap is well below mu_2, so INCONSISTENT DATA is the main cause;"
                          " check 1-lambda_3 and per-edge residuals rather than adding edges.\n"
                        : " -- gap tracks mu_2, so the GRAPH is the limit; add overlap between"
                          " weakly connected clusters or raise a trusted bridge edge's"
                          " confidence.\n");
    }

    // Undo the normalization: the blocks of Y are then proportional to R_p^T Q for a common but
    // unknown gauge Q.
    Eigen::MatrixXd Y(dim, 3);
    for (int k = 0; k < n_aug; ++k) {
      Y.middleRows(3 * k, 3) = d_inv_sqrt[k] * Z.middleRows(3 * k, 3);
    }

    // ---------- Step 4: fix the global reflection, then round onto SO(3) ----------
    // Q is only determined up to sign conventions and may have det -1, in which case EVERY block
    // has a negative determinant. Flipping one column of Y flips all of them at once. Blocks whose
    // sign then still disagrees with the component are genuinely inconsistent nodes.
    //
    // The vote runs over the REAL nodes only: the virtual node's block comes from a rank-one term,
    // so its determinant carries no orientation information and must not sway the decision.
    std::vector<double> dets(n);
    int sign_sum = 0;
    for (int k = 0; k < n; ++k) {
      dets[k] = Eigen::Matrix3d(Y.middleRows(3 * (voff + k), 3)).determinant();
      sign_sum += dets[k] < 0 ? -1 : 1;
    }
    if (sign_sum < 0) {
      Y.col(2) *= -1;
      for (int k = 0; k < n; ++k) {
        dets[k] = -dets[k];
      }
    }

    for (int k = 0; k < n; ++k) {
      // Y's blocks carry R_p^T, hence the transpose on the way out.
      const Eigen::Matrix3d Rp = projectToSO3(Eigen::Matrix3d(Y.middleRows(3 * (voff + k), 3)));
      result.rotations[nodes[k]] = Rp.transpose();
      result.valid[nodes[k]] = true;
      result.reliable[nodes[k]] = dets[k] > 0;
    }

    // ---------- Step 5: fix the gauge ----------
    // The cost is invariant under R_p -> Q R_p, so any global Q is free. Two choices:
    //
    //  * With gravity, pick the Q that stands the component upright. This is a pure gauge change --
    //    every relative rotation R_p^T R_q is untouched, exactly -- and it is what makes the
    //    per-node tilt below meaningful at all. Without it the frame is arbitrary and even perfect
    //    nodes report large "tilt".
    //  * Otherwise keep the historical convention: pin the lowest-indexed node to the identity, so
    //    the output is deterministic rather than drifting with the eigenbasis.
    Eigen::Matrix3d gauge = result.rotations[nodes[0]].transpose();
    if (gravity_count > 0) {
      Eigen::Vector3d summed = Eigen::Vector3d::Zero();
      for (int k = 0; k < n; ++k) {
        const Eigen::Vector3d& g = gravity[nodes[k]];
        if (g.squaredNorm() > 0) {
          summed += result.rotations[nodes[k]] * g;
        }
      }
      if (summed.norm() > 1e-12) {
        gauge = minimalRotation(summed.normalized(), world_up);
        result.gauge_upright[comp] = true;
      }
    }
    for (int k = 0; k < n; ++k) {
      result.rotations[nodes[k]] = gauge * result.rotations[nodes[k]];
    }

    // Residual tilt per node, meaningful only now that the gauge is fixed. An outlier near pi is an
    // upside-down scan; gravity cannot repair it (yaw is unconstrained) but it does reveal it.
    for (int k = 0; k < n; ++k) {
      const Eigen::Vector3d& g = gravity[nodes[k]];
      if (g.squaredNorm() > 0) {
        const double cs = (result.rotations[nodes[k]] * g).dot(world_up);
        result.tilt_error[nodes[k]] = std::acos(std::max(-1.0, std::min(1.0, cs)));
      }
    }
  }

  return result;
}
