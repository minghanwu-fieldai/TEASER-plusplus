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

  // Weighted point cost of a component, evaluated on component-local pose arrays (index = local id).
  auto localCost = [&](int comp, const std::vector<Eigen::Matrix3d>& Rl,
                       const std::vector<Eigen::Vector3d>& tl) {
    double cost = 0;
    for (const ReducedEdge* r : comp_edges[comp]) {
      const int lp = local_id[r->p];
      const int lq = local_id[r->q];
      const auto lhs = (Rl[lp] * (*r->p_pts)).colwise() + tl[lp];
      const auto rhs = (Rl[lq] * (*r->q_pts)).colwise() + tl[lq];
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
    // 6-DoF state block [xi (rotation) | dt (translation)], laid out at 6*(local_id - 1). The
    // current accepted estimate lives in the local arrays; result.* is only written at the end, so
    // a rejected trial never touches the returned poses.
    const int dim = 6 * (n - 1);
    std::vector<Eigen::Matrix3d> Rloc(n);
    std::vector<Eigen::Vector3d> tloc(n);
    for (int k = 0; k < n; ++k) {
      Rloc[k] = result.rotations[nodes[k]];
      tloc[k] = result.translations[nodes[k]];
    }
    double cur_cost = localCost(comp, Rloc, tloc);
    double lambda = params.lambda_init;

    for (int iter = 0; iter < params.max_iterations; ++iter) {
      // --- Gauss-Newton normal equations at the current estimate (undamped). ---
      std::vector<Eigen::Triplet<double>> base;
      base.reserve(comp_edges[comp].size() * 4 * 36);
      Eigen::VectorXd g = Eigen::VectorXd::Zero(dim);
      Eigen::VectorXd diagH = Eigen::VectorXd::Zero(dim);

      for (const ReducedEdge* r : comp_edges[comp]) {
        const int lp = local_id[r->p];
        const int lq = local_id[r->q];
        const Eigen::Matrix3d& Rp = Rloc[lp];
        const Eigen::Matrix3d& Rq = Rloc[lq];
        const Eigen::Vector3d& tp = tloc[lp];
        const Eigen::Vector3d& tq = tloc[lq];

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

          const bool p_free = lp > 0;
          const bool q_free = lq > 0;
          const int op = 6 * (lp - 1);
          const int oq = 6 * (lq - 1);

          auto addBlock = [&](int off_r, const Eigen::Matrix<double, 3, 6>& Jr, int off_c,
                              const Eigen::Matrix<double, 3, 6>& Jc) {
            const Eigen::Matrix<double, 6, 6> blk = omega * Jr.transpose() * Jc;
            for (int a2 = 0; a2 < 6; ++a2) {
              for (int b2 = 0; b2 < 6; ++b2) {
                base.emplace_back(off_r + a2, off_c + b2, blk(a2, b2));
              }
            }
          };

          if (p_free) {
            addBlock(op, Jp, op, Jp);
            g.segment<6>(op) += omega * Jp.transpose() * res;
            diagH.segment<6>(op) += omega * Jp.colwise().squaredNorm().transpose();
          }
          if (q_free) {
            addBlock(oq, Jq, oq, Jq);
            g.segment<6>(oq) += omega * Jq.transpose() * res;
            diagH.segment<6>(oq) += omega * Jq.colwise().squaredNorm().transpose();
          }
          if (p_free && q_free) {
            addBlock(op, Jp, oq, Jq);
            addBlock(oq, Jq, op, Jp);
          }
        }
      }
      for (int d = 0; d < dim; ++d) {
        diagH(d) = std::max(diagH(d), params.min_diagonal);
      }

      // --- Levenberg-Marquardt inner loop: raise lambda until a step lowers the cost. ---
      bool accepted = false;
      double rel_decrease = 0;
      double max_step = 0;
      while (lambda <= params.lambda_max) {
        std::vector<Eigen::Triplet<double>> triplets = base;
        for (int d = 0; d < dim; ++d) {
          triplets.emplace_back(d, d, lambda * diagH(d));
        }
        Eigen::SparseMatrix<double> H(dim, dim);
        H.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) {
          lambda *= params.lambda_factor;
          continue;
        }
        const Eigen::VectorXd step = solver.solve(-g);
        if (solver.info() != Eigen::Success) {
          lambda *= params.lambda_factor;
          continue;
        }

        // Trial retraction into scratch arrays; result is untouched unless we accept.
        std::vector<Eigen::Matrix3d> Rtry = Rloc;
        std::vector<Eigen::Vector3d> ttry = tloc;
        for (int k = 1; k < n; ++k) {
          const int off = 6 * (k - 1);
          Rtry[k] = Rloc[k] * so3Exp(step.segment<3>(off));
          ttry[k] = tloc[k] + step.segment<3>(off + 3);
        }
        const double trial_cost = localCost(comp, Rtry, ttry);

        if (trial_cost < cur_cost) {
          rel_decrease = cur_cost > 0 ? (cur_cost - trial_cost) / cur_cost : cur_cost - trial_cost;
          max_step = step.cwiseAbs().maxCoeff();
          Rloc.swap(Rtry);
          tloc.swap(ttry);
          cur_cost = trial_cost;
          lambda = std::max(lambda / params.lambda_factor, params.lambda_min);
          accepted = true;
          result.iterations[comp] = iter + 1;
          break;
        }
        lambda *= params.lambda_factor; // overshoot: damp harder and retry
      }

      // No damping level improved the cost -> at a local minimum (or numerically stuck). Stop; the
      // best estimate so far is already in Rloc/tloc.
      if (!accepted) {
        break;
      }
      if (max_step < params.step_tol || rel_decrease < params.cost_tol) {
        break;
      }
    }

    // Commit the best (monotonically non-worsening) estimate for this component.
    for (int k = 0; k < n; ++k) {
      result.rotations[nodes[k]] = Rloc[k];
      result.translations[nodes[k]] = tloc[k];
    }
  }

  // Diagnostics: how far refinement moved the poses, averaged over the refined nodes.
  double t_change_sum = 0.0;
  double r_change_sum = 0.0;
  int refined_count = 0;
  for (int i = 0; i < num_nodes; ++i) {
    if (!result.valid[i]) {
      continue;
    }
    t_change_sum += (result.translations[i] - t_init[i]).norm();
    const double c = ((R_init[i].transpose() * result.rotations[i]).trace() - 1.0) / 2.0;
    r_change_sum += std::acos(std::max(-1.0, std::min(1.0, c)));
    ++refined_count;
  }
  if (refined_count > 0) {
    result.avg_translation_change = t_change_sum / refined_count;
    result.avg_rotation_change = r_change_sum / refined_count;
  }

  return result;
}
