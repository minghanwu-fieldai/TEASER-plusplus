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
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
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

/** An edge that survived preprocessing, reduced to just what synchronization needs. */
struct ReducedEdge {
  int p;
  int q;
  double c;
  Eigen::Matrix3d Rhat; // approximately R_q^T R_p

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Top-3 eigenspace of the symmetric matrix Bn (dimension dim), plus the gap between the 3rd and 4th
 * largest eigenvalues (-1 when the matrix is too small for a 4th eigenvalue).
 *
 * The three returned columns are an arbitrary basis of that eigenspace; their order does not matter
 * because any change of basis is absorbed into the global gauge.
 */
bool topThreeEigenspaceDense(const Eigen::SparseMatrix<double>& Bn, int dim, Eigen::MatrixXd* Z,
                             double* gap) {
  const Eigen::MatrixXd dense = Eigen::MatrixXd(Bn);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(dense);
  if (eig.info() != Eigen::Success) {
    return false;
  }
  // Eigen sorts eigenvalues ascending, so the top-3 eigenspace is the last three columns.
  *Z = eig.eigenvectors().rightCols(3);
  const Eigen::VectorXd& asc = eig.eigenvalues();
  *gap = dim >= 4 ? asc(dim - 3) - asc(dim - 4) : -1.0;
  return true;
}

bool topThreeEigenspaceSparse(const Eigen::SparseMatrix<double>& Bn, int dim, Eigen::MatrixXd* Z,
                              double* gap) {
  const int nev = std::min(6, dim - 1);
  if (nev < 3) {
    return false;
  }
  const int ncv = std::min(dim, std::max(2 * nev + 1, 20));
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
  // For LARGEST_ALGE, Spectra returns eigenvalues in descending order.
  *Z = eigs.eigenvectors().leftCols(3);
  const Eigen::VectorXd vals = eigs.eigenvalues();
  *gap = (nconv >= 4 && vals.size() >= 4) ? vals(2) - vals(3) : -1.0;
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
                             const std::vector<double>* node_noise_bounds) {
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

  result.rotations.assign(num_nodes, Eigen::Matrix3d::Identity());
  if (use_initial) {
    result.rotations = *initial;
  }
  result.valid.assign(num_nodes, false);
  result.reliable.assign(num_nodes, false);
  result.component.assign(num_nodes, -1);

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
    const int dim = 3 * n;

    // A component is built from edges, so it always holds at least two nodes. Guard anyway: a lone
    // node's rotation is pure gauge, so the identity is correct.
    if (n < 2) {
      result.rotations[nodes[0]] = Eigen::Matrix3d::Identity();
      result.valid[nodes[0]] = true;
      result.reliable[nodes[0]] = true;
      continue;
    }

    // ---------- Step 2: assemble the normalized synchronization matrix ----------
    // Node degrees d[p] = sum of incident edge confidences.
    std::vector<double> d(n, 0.0);
    for (const ReducedEdge* r : comp_edges[comp]) {
      d[local_id[r->p]] += r->c;
      d[local_id[r->q]] += r->c;
    }
    std::vector<double> d_inv_sqrt(n);
    for (int k = 0; k < n; ++k) {
      d_inv_sqrt[k] = d[k] > 0 ? 1.0 / std::sqrt(d[k]) : 0.0;
    }

    // Bn = D^{-1/2} B D^{-1/2} where B has block (q,p) = c * Rhat and block (p,q) = c * Rhat^T.
    // The D^{-1/2} scaling is folded straight into the triplets rather than materialized, so each
    // entry of block (q,p) is scaled by 1/sqrt(d_q d_p). Spectral radius <= 1 by construction.
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(comp_edges[comp].size() * 18);
    for (const ReducedEdge* r : comp_edges[comp]) {
      const int lp = local_id[r->p];
      const int lq = local_id[r->q];
      const double s = r->c * d_inv_sqrt[lp] * d_inv_sqrt[lq];
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const double v = s * r->Rhat(a, b);
          triplets.emplace_back(3 * lq + a, 3 * lp + b, v);
          triplets.emplace_back(3 * lp + b, 3 * lq + a, v);
        }
      }
    }
    Eigen::SparseMatrix<double> Bn(dim, dim);
    // setFromTriplets sums duplicates, so parallel edges on the same node pair accumulate rather
    // than one of them silently winning.
    Bn.setFromTriplets(triplets.begin(), triplets.end());

    // ---------- Step 3: top-3 eigenspace ----------
    Eigen::MatrixXd Z;
    double gap = -1.0;
    bool ok = false;
    if (dim > params.dense_max_dim) {
      ok = topThreeEigenspaceSparse(Bn, dim, &Z, &gap);
      if (!ok) {
        std::cerr << "[teaser::rotation_sync] Warning: iterative eigensolver failed on component "
                  << comp << "; falling back to a dense decomposition.\n";
      }
    }
    if (!ok) {
      ok = topThreeEigenspaceDense(Bn, dim, &Z, &gap);
    }
    if (!ok) {
      std::cerr << "[teaser::rotation_sync] Warning: eigendecomposition failed on component "
                << comp << "; leaving its nodes unestimated.\n";
      continue;
    }
    result.spectral_gap[comp] = gap;
    if (gap >= 0 && gap < params.gap_threshold) {
      std::cerr << "[teaser::rotation_sync] Warning: component " << comp << " has spectral gap "
                << gap << " (< " << params.gap_threshold
                << "); weak connectivity, rotations are poorly determined.\n";
    }

    // Undo the normalization: the blocks of Y are then proportional to R_p^T Q for a common but
    // unknown gauge Q.
    Eigen::MatrixXd Y(dim, 3);
    for (int k = 0; k < n; ++k) {
      Y.middleRows(3 * k, 3) = d_inv_sqrt[k] * Z.middleRows(3 * k, 3);
    }

    // ---------- Step 4: fix the global reflection, then round onto SO(3) ----------
    // Q is only determined up to sign conventions and may have det -1, in which case EVERY block
    // has a negative determinant. Flipping one column of Y flips all of them at once. Blocks whose
    // sign then still disagrees with the component are genuinely inconsistent nodes.
    std::vector<double> dets(n);
    int sign_sum = 0;
    for (int k = 0; k < n; ++k) {
      dets[k] = Eigen::Matrix3d(Y.middleRows(3 * k, 3)).determinant();
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
      const Eigen::Matrix3d Rp = projectToSO3(Eigen::Matrix3d(Y.middleRows(3 * k, 3)));
      result.rotations[nodes[k]] = Rp.transpose();
      result.valid[nodes[k]] = true;
      result.reliable[nodes[k]] = dets[k] > 0;
    }

    // ---------- Step 5: fix the gauge ----------
    // The cost is invariant under R_p -> Q R_p, so left-multiplying by the anchor's inverse is free.
    // Pinning the lowest-indexed node to the identity keeps the output deterministic across calls
    // instead of drifting with whatever eigenbasis the solver happened to return.
    const Eigen::Matrix3d anchor_inv = result.rotations[nodes[0]].transpose();
    for (int k = 0; k < n; ++k) {
      result.rotations[nodes[k]] = anchor_inv * result.rotations[nodes[k]];
    }
  }

  return result;
}
