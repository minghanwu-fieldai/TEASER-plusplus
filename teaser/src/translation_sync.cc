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
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "sync_components.h"
#include "translation_sync.h"

namespace {

/** An edge that survived preprocessing, reduced to the two statistics the solve needs. */
struct ReducedEdge {
  int p;
  int q;
  double weight;      // Laplacian edge weight: total correspondence mass, scaled by 1/sigma^2
  Eigen::Vector3d d;  // target for t_p - t_q

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace

double teaser::pointResidualNoiseBound(double delta_p, double delta_q) {
  return delta_p + delta_q;
}

teaser::TranslationSyncResult teaser::synchronizeTranslations(
    int num_nodes, const std::vector<teaser::TranslationSyncEdge>& edges,
    const std::vector<Eigen::Matrix3d>& rotations, const teaser::TranslationSyncParams& params,
    const std::vector<Eigen::Vector3d>* initial, const std::vector<double>* node_noise_bounds,
    const std::vector<std::pair<int, int>>* pairs_of_interest) {
  teaser::TranslationSyncResult result;
  if (num_nodes <= 0) {
    return result;
  }
  if (static_cast<int>(rotations.size()) != num_nodes) {
    std::cerr << "[teaser::translation_sync] Error: rotations has " << rotations.size()
              << " entries but there are " << num_nodes << " nodes; returning nothing.\n";
    return result;
  }

  const bool use_initial = initial != nullptr && static_cast<int>(initial->size()) == num_nodes;
  const bool use_noise_bounds =
      node_noise_bounds != nullptr && static_cast<int>(node_noise_bounds->size()) == num_nodes;
  if (node_noise_bounds != nullptr && !use_noise_bounds) {
    std::cerr << "[teaser::translation_sync] Warning: node_noise_bounds has "
              << node_noise_bounds->size() << " entries but there are " << num_nodes
              << " nodes; ignoring it.\n";
  }

  result.translations.assign(num_nodes, Eigen::Vector3d::Zero());
  if (use_initial) {
    result.translations = *initial;
  }
  result.valid.assign(num_nodes, false);
  result.component.assign(num_nodes, -1);

  // ---------- Step 0: reduce each edge to (weight, target) ----------
  std::vector<ReducedEdge> kept;
  kept.reserve(edges.size());
  for (const auto& e : edges) {
    if (e.p < 0 || e.q < 0 || e.p >= num_nodes || e.q >= num_nodes || e.p == e.q) {
      continue;
    }
    const Eigen::Index m = e.p_pts.cols();
    if (m == 0 || e.q_pts.cols() != m || e.w.cols() != m) {
      continue;
    }
    if (!(e.confidence > 0)) {
      continue;
    }
    const double mass = e.w.sum();
    if (mass < params.min_edge_mass) {
      continue; // GNC rejected every correspondence on this edge
    }

    double weight = e.confidence * mass;
    if (use_noise_bounds) {
      // Raw points, so each scan contributes its per-point bound once -- no factor of 2.
      const double sigma =
          teaser::pointResidualNoiseBound((*node_noise_bounds)[e.p], (*node_noise_bounds)[e.q]);
      if (!(sigma > 0)) {
        continue; // both scans declared noiseless; the weight would be infinite
      }
      weight /= sigma * sigma;
    }

    // Collapsing the edge to its weighted centroids is exact: the inner sum depends on t_p and t_q
    // only through their difference, so sum_j w_j ||(t_p - t_q) - c_j||^2 equals
    // mass * ||(t_p - t_q) - d||^2 up to a constant. Note the centroids use the raw w, not the
    // noise-scaled weight -- sigma is constant across the edge, so it cancels out of the mean.
    ReducedEdge r;
    r.p = e.p;
    r.q = e.q;
    r.weight = weight;
    const Eigen::Vector3d p_bar = (e.p_pts * e.w.transpose()) / mass;
    const Eigen::Vector3d q_bar = (e.q_pts * e.w.transpose()) / mass;
    r.d = rotations[e.q] * q_bar - rotations[e.p] * p_bar;
    kept.push_back(r);
  }

  // ---------- Step 1: connected components over the surviving edges ----------
  std::vector<std::pair<int, int>> kept_pairs;
  kept_pairs.reserve(kept.size());
  for (const auto& r : kept) {
    kept_pairs.emplace_back(r.p, r.q);
  }
  const teaser::SyncComponents comps = teaser::labelSyncComponents(num_nodes, kept_pairs);
  result.component = comps.component;
  result.num_components = comps.num_components;
  result.residual_sigma.assign(result.num_components, -1.0);
  result.fiedler_value.assign(result.num_components, -1.0);
  if (result.num_components == 0) {
    return result;
  }
  if (result.num_components > 1) {
    std::cerr << "[teaser::translation_sync] Warning: graph has " << result.num_components
              << " connected components; synchronizing each independently.\n";
  }

  // Component-local ids. comps.nodes[c] is ascending, so local id 0 is the anchor.
  std::vector<int> local_id(num_nodes, -1);
  for (const auto& nodes : comps.nodes) {
    for (size_t k = 0; k < nodes.size(); ++k) {
      local_id[nodes[k]] = static_cast<int>(k);
    }
  }
  std::vector<std::vector<const ReducedEdge*>> comp_edges(result.num_components);
  for (const auto& r : kept) {
    comp_edges[result.component[r.p]].push_back(&r);
  }

  for (int comp = 0; comp < result.num_components; ++comp) {
    const std::vector<int>& nodes = comps.nodes[comp];
    const int n = static_cast<int>(nodes.size());

    // A component is built from edges, so it always holds at least two nodes. Guard anyway.
    if (n < 2) {
      result.translations[nodes[0]] = Eigen::Vector3d::Zero();
      result.valid[nodes[0]] = true;
      continue;
    }

    // ---------- Step 2: assemble the grounded Laplacian and the right-hand sides ----------
    // The anchor (local id 0) is pinned to zero, so its row and column are simply never emitted.
    // That leaves the grounded Laplacian: still sparse, and positive definite for a connected
    // component -- unlike L + ones(n,n)/n, whose dense rank-1 term would destroy the sparsity for
    // a zero-mean gauge that pinning the anchor discards anyway.
    const int dim = n - 1;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(comp_edges[comp].size() * 4);
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(dim, 3);
    for (const ReducedEdge* r : comp_edges[comp]) {
      const int lp = local_id[r->p];
      const int lq = local_id[r->q];
      const double weight = r->weight;
      if (lp > 0) {
        triplets.emplace_back(lp - 1, lp - 1, weight);
        rhs.row(lp - 1) += weight * r->d.transpose();
      }
      if (lq > 0) {
        triplets.emplace_back(lq - 1, lq - 1, weight);
        rhs.row(lq - 1) -= weight * r->d.transpose(); // d[q,p] == -d[p,q]
      }
      if (lp > 0 && lq > 0) {
        triplets.emplace_back(lp - 1, lq - 1, -weight);
        triplets.emplace_back(lq - 1, lp - 1, -weight);
      }
    }
    Eigen::SparseMatrix<double> L(dim, dim);
    // setFromTriplets sums duplicates, so parallel edges accumulate rather than one of them
    // silently winning.
    L.setFromTriplets(triplets.begin(), triplets.end());

    // ---------- Step 3: one factorization, three right-hand sides ----------
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(L);
    if (solver.info() != Eigen::Success) {
      std::cerr << "[teaser::translation_sync] Warning: factorization failed on component " << comp
                << "; leaving its nodes unestimated.\n";
      continue;
    }
    const Eigen::MatrixXd sol = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
      std::cerr << "[teaser::translation_sync] Warning: solve failed on component " << comp
                << "; leaving its nodes unestimated.\n";
      continue;
    }

    result.translations[nodes[0]] = Eigen::Vector3d::Zero(); // anchor pins the gauge
    result.valid[nodes[0]] = true;
    for (int k = 1; k < n; ++k) {
      result.translations[nodes[k]] = sol.row(k - 1).transpose();
      result.valid[nodes[k]] = true;
    }

    if (!params.compute_diagnostics) {
      continue;
    }

    // ---------- Step 4: diagnostics ----------
    // Residual scale. The reduced residual measures how much the edges disagree with each other;
    // the within-edge scatter is the constant the reduction dropped and says nothing about the fit.
    const int num_edges = static_cast<int>(comp_edges[comp].size());
    const int dof = 3 * (num_edges - (n - 1));
    if (dof > 0) {
      double weighted_residual_sum = 0;
      for (const ReducedEdge* r : comp_edges[comp]) {
        const Eigen::Vector3d res =
            (result.translations[r->p] - result.translations[r->q]) - r->d;
        weighted_residual_sum += r->weight * res.squaredNorm();
      }
      result.residual_sigma[comp] = std::sqrt(weighted_residual_sum / dof);
    }

    // Fiedler value of the FULL (ungrounded) Laplacian: its smallest eigenvalue is 0 by
    // construction, so lambda_2 is the second smallest.
    if (n <= params.fiedler_max_nodes) {
      Eigen::MatrixXd full = Eigen::MatrixXd::Zero(n, n);
      for (const ReducedEdge* r : comp_edges[comp]) {
        const int lp = local_id[r->p];
        const int lq = local_id[r->q];
        full(lp, lp) += r->weight;
        full(lq, lq) += r->weight;
        full(lp, lq) -= r->weight;
        full(lq, lp) -= r->weight;
      }
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(full);
      if (eig.info() == Eigen::Success) {
        result.fiedler_value[comp] = eig.eigenvalues()(1); // ascending
      }
    }

    // Relative-translation uncertainty for the requested pairs. Effective resistance comes from
    // reusing the factorization above; forming pinv(L) would be dense and O(n^3).
    if (pairs_of_interest != nullptr && result.residual_sigma[comp] >= 0) {
      for (const auto& pr : *pairs_of_interest) {
        if (pr.first < 0 || pr.second < 0 || pr.first >= num_nodes || pr.second >= num_nodes) {
          continue;
        }
        if (result.component[pr.first] != comp || result.component[pr.second] != comp ||
            pr.first == pr.second) {
          continue;
        }
        // e_p - e_q restricted to the grounded indices; an anchor entry simply drops out, which is
        // consistent because the anchor's translation is exactly zero.
        Eigen::VectorXd v = Eigen::VectorXd::Zero(dim);
        const int lp = local_id[pr.first];
        const int lq = local_id[pr.second];
        if (lp > 0) {
          v(lp - 1) += 1.0;
        }
        if (lq > 0) {
          v(lq - 1) -= 1.0;
        }
        const double r_eff = v.dot(solver.solve(v));
        const auto key = std::make_pair(std::min(pr.first, pr.second),
                                        std::max(pr.first, pr.second));
        result.pair_sigma[key] =
            r_eff > 0 ? result.residual_sigma[comp] * std::sqrt(r_eff) : 0.0;
      }
    }
  }

  return result;
}
