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
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "pose_refine.h"
#include "sync_components.h"

namespace {

// Skew-symmetric (hat) map. Local copy on purpose: teaser::hatmap in linalg.h is a non-inline free
// function already defined in certification.cc's translation unit, so including it here would be a
// duplicate-symbol link error.
Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

// SO(3) exponential of a rotation vector, small-angle guarded.
Eigen::Matrix3d so3Exp(const Eigen::Vector3d& v) {
  const double theta = v.norm();
  if (theta < 1e-12) {
    // First-order retraction; re-orthonormalized so the result is exactly a rotation.
    Eigen::Matrix3d approx = Eigen::Matrix3d::Identity() + hat(v);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(approx, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
  }
  return Eigen::AngleAxisd(theta, v / theta).toRotationMatrix();
}

/** An edge that survived preprocessing, reduced to what the solve needs. */
struct ReducedEdge {
  int p;
  int q;
  double edge_weight;
  const Eigen::Matrix<double, 3, Eigen::Dynamic>* p_pts;
  const Eigen::Matrix<double, 3, Eigen::Dynamic>* q_pts;
  const Eigen::Matrix<double, 1, Eigen::Dynamic>* w;
};

} // namespace

teaser::PoseRefineResult
teaser::refinePoses(int num_nodes, const std::vector<teaser::PoseRefineEdge>& edges,
                    const std::vector<Eigen::Matrix3d>& R_init,
                    const std::vector<Eigen::Vector3d>& t_init,
                    const teaser::PoseRefineParams& params) {
  teaser::PoseRefineResult result;
  if (num_nodes <= 0) {
    return result;
  }
  if (static_cast<int>(R_init.size()) != num_nodes ||
      static_cast<int>(t_init.size()) != num_nodes) {
    std::cerr << "[teaser::pose_refine] Error: R_init/t_init size does not match num_nodes; "
                 "returning nothing.\n";
    return result;
  }

  result.rotations = R_init;
  result.translations = t_init;
  result.valid.assign(num_nodes, false);
  result.component.assign(num_nodes, -1);

  // ---------- Validate edges ----------
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
    if (!(e.edge_weight > 0)) {
      continue;
    }
    kept.push_back({e.p, e.q, e.edge_weight, &e.p_pts, &e.q_pts, &e.w});
  }

  // ---------- Connected components (shared with the sync stages) ----------
  std::vector<std::pair<int, int>> kept_pairs;
  kept_pairs.reserve(kept.size());
  for (const auto& r : kept) {
    kept_pairs.emplace_back(r.p, r.q);
  }
  const teaser::SyncComponents comps = teaser::labelSyncComponents(num_nodes, kept_pairs);
  result.component = comps.component;
  result.num_components = comps.num_components;
  result.iterations.assign(result.num_components, 0);
  if (result.num_components == 0) {
    return result;
  }

  // Every node touched by an edge is valid; its pose is refined below (edgeless nodes stay at init).
  for (int i = 0; i < num_nodes; ++i) {
    if (result.component[i] >= 0) {
      result.valid[i] = true;
    }
  }

  // Component-local ids. comps.nodes[c] is ascending, so local id 0 is the anchor -- held fixed.
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

  auto componentCost = [&](const std::vector<const ReducedEdge*>& es) {
    double cost = 0;
    for (const ReducedEdge* r : es) {
      const auto lhs = (result.rotations[r->p] * (*r->p_pts)).colwise() + result.translations[r->p];
      const auto rhs = (result.rotations[r->q] * (*r->q_pts)).colwise() + result.translations[r->q];
      const Eigen::Matrix<double, 3, Eigen::Dynamic> diff = lhs - rhs;
      cost += r->edge_weight * (diff.colwise().squaredNorm().array() * r->w->array()).sum();
    }
    return cost;
  };

  for (int comp = 0; comp < result.num_components; ++comp) {
    const std::vector<int>& nodes = comps.nodes[comp];
    const int n = static_cast<int>(nodes.size());
    if (n < 2) {
      continue; // lone node: nothing to refine (its pose is pure gauge)
    }

    // Free nodes are the component's nodes minus the anchor (local id 0). Each free node has a
    // 6-DoF state block [xi (rotation) | dt (translation)], laid out at 6*(local_id - 1).
    const int dim = 6 * (n - 1);
    double prev_cost = componentCost(comp_edges[comp]);

    for (int iter = 0; iter < params.max_iterations; ++iter) {
      std::vector<Eigen::Triplet<double>> triplets;
      triplets.reserve(comp_edges[comp].size() * 4 * 144);
      Eigen::VectorXd g = Eigen::VectorXd::Zero(dim);

      for (const ReducedEdge* r : comp_edges[comp]) {
        const int lp = local_id[r->p];
        const int lq = local_id[r->q];
        const Eigen::Matrix3d& Rp = result.rotations[r->p];
        const Eigen::Matrix3d& Rq = result.rotations[r->q];
        const Eigen::Vector3d& tp = result.translations[r->p];
        const Eigen::Vector3d& tq = result.translations[r->q];

        for (Eigen::Index j = 0; j < r->p_pts->cols(); ++j) {
          const double omega = r->edge_weight * (*r->w)(j);
          if (omega <= 0) {
            continue;
          }
          const Eigen::Vector3d a = r->p_pts->col(j);
          const Eigen::Vector3d b = r->q_pts->col(j);
          const Eigen::Vector3d res = (Rp * a + tp) - (Rq * b + tq);

          // 3x6 Jacobian block per endpoint: [ -R [pt]x | +-I ].
          Eigen::Matrix<double, 3, 6> Jp, Jq;
          Jp.leftCols<3>() = -Rp * hat(a);
          Jp.rightCols<3>() = Eigen::Matrix3d::Identity();
          Jq.leftCols<3>() = Rq * hat(b);
          Jq.rightCols<3>() = -Eigen::Matrix3d::Identity();

          // Accumulate omega * J^T J into H and omega * J^T res into g, dropping the anchor's block
          // (local id 0). free(l) maps a free local id to its 6-block offset.
          const bool p_free = lp > 0;
          const bool q_free = lq > 0;
          const int op = 6 * (lp - 1);
          const int oq = 6 * (lq - 1);

          auto addBlock = [&](int off_r, const Eigen::Matrix<double, 3, 6>& Jr, int off_c,
                              const Eigen::Matrix<double, 3, 6>& Jc) {
            const Eigen::Matrix<double, 6, 6> blk = omega * Jr.transpose() * Jc;
            for (int a2 = 0; a2 < 6; ++a2) {
              for (int b2 = 0; b2 < 6; ++b2) {
                triplets.emplace_back(off_r + a2, off_c + b2, blk(a2, b2));
              }
            }
          };

          if (p_free) {
            addBlock(op, Jp, op, Jp);
            g.segment<6>(op) += omega * Jp.transpose() * res;
          }
          if (q_free) {
            addBlock(oq, Jq, oq, Jq);
            g.segment<6>(oq) += omega * Jq.transpose() * res;
          }
          if (p_free && q_free) {
            addBlock(op, Jp, oq, Jq);
            addBlock(oq, Jq, op, Jp);
          }
        }
      }

      // Tikhonov damping on the diagonal for rank safety.
      for (int d = 0; d < dim; ++d) {
        triplets.emplace_back(d, d, params.damping);
      }

      Eigen::SparseMatrix<double> H(dim, dim);
      H.setFromTriplets(triplets.begin(), triplets.end());
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(H);
      if (solver.info() != Eigen::Success) {
        std::cerr << "[teaser::pose_refine] Warning: factorization failed on component " << comp
                  << "; stopping refinement for it.\n";
        break;
      }
      const Eigen::VectorXd step = solver.solve(-g); // H delta = -g
      if (solver.info() != Eigen::Success) {
        std::cerr << "[teaser::pose_refine] Warning: solve failed on component " << comp
                  << "; stopping refinement for it.\n";
        break;
      }

      // Retract each free node: R <- R exp([xi]x), t <- t + dt.
      for (int k = 1; k < n; ++k) {
        const int off = 6 * (k - 1);
        const Eigen::Vector3d xi = step.segment<3>(off);
        const Eigen::Vector3d dt = step.segment<3>(off + 3);
        result.rotations[nodes[k]] = result.rotations[nodes[k]] * so3Exp(xi);
        result.translations[nodes[k]] += dt;
      }
      result.iterations[comp] = iter + 1;

      const double cost = componentCost(comp_edges[comp]);
      const double max_step = step.cwiseAbs().maxCoeff();
      const double rel_decrease =
          prev_cost > 0 ? (prev_cost - cost) / prev_cost : std::abs(prev_cost - cost);
      prev_cost = cost;
      if (max_step < params.step_tol || std::abs(rel_decrease) < params.cost_tol) {
        break;
      }
    }
  }

  return result;
}
