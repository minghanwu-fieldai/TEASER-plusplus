/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <vector>

#include <Eigen/Core>

namespace teaser {

/**
 * One edge (p, q) of the pose-refinement graph.
 *
 * Unlike the rotation-synchronization stage these are RAW points, not TIMs -- refinement optimizes
 * rotation and translation jointly, so translation must stay in the residual. Correspondences are
 * pre-matched column-wise: p_pts.col(j) and q_pts.col(j) are the same world point seen in the two
 * scans' local frames, so at the optimum R_p * p_pts.col(j) + t_p == R_q * q_pts.col(j) + t_q.
 */
struct PoseRefineEdge {
  /** Index of the first endpoint. */
  int p = -1;
  /** Index of the second endpoint. */
  int q = -1;
  /** Raw points in p's local frame, 3-by-m. */
  Eigen::Matrix<double, 3, Eigen::Dynamic> p_pts;
  /** Raw points in q's local frame, 3-by-m (column j <-> p_pts column j). */
  Eigen::Matrix<double, 3, Eigen::Dynamic> q_pts;
  /** Per-correspondence weights, 1-by-m -- the GNC line-process weights, as in the sync stages. */
  Eigen::Matrix<double, 1, Eigen::Dynamic> w;
  /**
   * Whole-edge scale multiplying every correspondence of this edge, i.e. confidence / sigma^2.
   * A uniform factor across all edges has no effect on the minimizer (it scales the whole cost),
   * so under a single global noise bound this can be left at 1.
   */
  double edge_weight = 1.0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Tuning knobs for refinePoses.
 */
struct PoseRefineParams {
  /** Maximum Gauss-Newton iterations per connected component. 0 disables refinement (returns the
   * initialization unchanged) -- useful as a control in tests. */
  int max_iterations = 10;
  /** Stop when the largest element of the pose increment falls below this. */
  double step_tol = 1e-10;
  /** Stop when the relative cost decrease between iterations falls below this. */
  double cost_tol = 1e-12;
  /**
   * Tikhonov damping added to the diagonal of the normal-equations matrix. Guards against a
   * rank-deficient Hessian on near-degenerate geometry; small enough to be negligible otherwise.
   */
  double damping = 1e-9;
};

/**
 * Result of a pose refinement.
 */
struct PoseRefineResult {
  /** Refined rotation per node (local -> world). Unchanged from the init for edgeless nodes. */
  std::vector<Eigen::Matrix3d> rotations;
  /** Refined translation per node (local -> world). Unchanged from the init for edgeless nodes. */
  std::vector<Eigen::Vector3d> translations;
  /** False for a node that had no edge, so its pose was not refined. */
  std::vector<bool> valid;
  /** Connected-component id per node over the edges; -1 for a node with no edge. */
  std::vector<int> component;
  /** Number of connected components. */
  int num_components = 0;
  /** Gauss-Newton iterations actually run, per component (diagnostic). */
  std::vector<int> iterations;
};

/**
 * Locally refine all node poses by joint Gauss-Newton on the raw-point objective.
 *
 * Minimizes, per connected component,
 *
 *     sum_(p,q) sum_j  w_j * edge_weight * || (R_p a_j + t_p) - (R_q b_j + t_q) ||^2
 *
 * over R_i in SO(3) and t_i in R^3, seeded by (R_init, t_init). This is the true measurement cost,
 * so it recovers the information the spectral rotation and Laplacian translation relaxations discard
 * (per-edge anisotropy) and removes the rotation->translation error coupling by solving both at
 * once. Point noise is i.i.d. isotropic, so uniform per-correspondence weighting is already ML.
 *
 * It is a LOCAL method: it polishes a good initialization and will not cross a 180-degree basin, so
 * it cannot repair a flipped input pose -- only reduce residual error near the seed.
 *
 * Gauge: each component's lowest-indexed node is held fixed at its initial pose (the same anchor the
 * synchronization stages pin, which is (I, 0) there), so the refined poses compose directly with a
 * downstream re-gauge.
 *
 * @param num_nodes [in] number of nodes; edges must reference 0..num_nodes-1
 * @param edges [in] the refinement edges with their weighted raw correspondences
 * @param R_init [in] per-node initial rotations; must be sized num_nodes
 * @param t_init [in] per-node initial translations; must be sized num_nodes
 * @param params [in] tuning knobs
 * @return the refined per-node poses plus validity and iteration diagnostics
 */
PoseRefineResult refinePoses(int num_nodes, const std::vector<PoseRefineEdge>& edges,
                             const std::vector<Eigen::Matrix3d>& R_init,
                             const std::vector<Eigen::Vector3d>& t_init,
                             const PoseRefineParams& params = PoseRefineParams());

} // namespace teaser
