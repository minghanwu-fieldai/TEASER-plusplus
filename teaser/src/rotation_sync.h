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
 * One edge (p, q) of the rotation synchronization graph.
 *
 * The correspondences MUST be translation-invariant measurements (TIMs) -- differences of point
 * pairs -- and NOT raw points. This edge contributes ||R_p * p_tims_j - R_q * q_tims_j||^2, a cost
 * with no translation term in it, so raw local-frame points would leave each scan's translation
 * sitting in the residual and bias the rotation. Concretely, with y_j = A x_j + t the cross
 * covariance picks up a spurious rank-1 term,
 *
 *     M = sum_j w_j x_j y_j^T = (sum_j w_j x_j x_j^T) A^T + (sum_j w_j x_j) t^T
 *
 * which is only harmless when t = 0. Differencing a pair of points cancels t exactly, so TIMs make
 * it so. This matches the pairwise pipeline, which feeds pruned_src_tims_ / pruned_dst_tims_ -- not
 * points -- into its GNC rotation solver.
 *
 * Subtracting centroids would kill the same term (by making sum_j w_j x_j = 0) but is deliberately
 * not the approach here: a weighted centroid is a global statistic that one gross outlier corrupts,
 * which would defeat the downstream outlier rejection, and under GNC the weights change every
 * iteration so the centroid would drift as mu anneals. TIMs are computed once and are individually
 * testable.
 *
 * Nothing in this header can enforce the requirement -- TIMs and points are both just 3-by-m
 * matrices -- so it is on the caller.
 *
 * Correspondences are pre-matched column-wise: p_tims.col(j) is a TIM in node p's local frame and
 * q_tims.col(j) is the TIM it matches in node q's local frame. Both describe the same world-frame
 * displacement, so at the optimum R_p * p_tims.col(j) == R_q * q_tims.col(j).
 */
struct RotationSyncEdge {
  /** Index of the first endpoint. */
  int p = -1;
  /** Index of the second endpoint. */
  int q = -1;
  /** TIM vectors expressed in p's local frame, 3-by-m. */
  Eigen::Matrix<double, 3, Eigen::Dynamic> p_tims;
  /** TIM vectors expressed in q's local frame, 3-by-m (column j <-> p_tims column j). */
  Eigen::Matrix<double, 3, Eigen::Dynamic> q_tims;
  /**
   * Per-correspondence weights, 1-by-m. These are the GNC line-process weights (times any
   * per-correspondence prior), so they change every GNC iteration.
   *
   * CAUTION for callers running an outer GNC loop: these only reshape the cross-covariance M
   * *within* this edge. Rhat is M's polar factor, which is invariant to scaling M, so scaling every
   * w on an edge by a common factor changes nothing -- an edge whose correspondences have all been
   * driven to near-zero weight still exerts its full influence. Between-edge influence comes solely
   * from `confidence`, so fold the surviving weight mass into it (`confidence *= w.sum()`) if the
   * graph is meant to be able to reject a coherently wrong edge.
   */
  Eigen::Matrix<double, 1, Eigen::Dynamic> w;
  /**
   * Whole-edge confidence c[p,q]: scales this edge's block in the synchronization matrix and its
   * contribution to the node degrees. Currently always left at 1; the hook exists so a real
   * per-edge confidence (overlap ratio, inlier fraction, ...) can be supplied later.
   *
   * When per-scan noise bounds are supplied to synchronizeRotations this is multiplied by
   * 1 / sigma_pq^2, so it stays a pure confidence and the noise model is kept separate.
   */
  double confidence = 1.0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Tuning knobs for synchronizeRotations.
 */
struct RotationSyncParams {
  /**
   * Edges whose total correspondence weight (sum_j w_j) falls below this are dropped before the
   * graph is split into components. An edge whose correspondences have all been rejected by GNC
   * carries no information, and keeping it would fabricate a constraint out of numerical noise.
   */
  double min_edge_mass = 1e-9;
  /**
   * Diagnostic threshold on the spectral gap. A gap below this means the graph barely determines a
   * unique rotation assignment (weak connectivity); a warning is emitted but the result is still
   * returned.
   */
  double gap_threshold = 1e-3;
  /**
   * Components whose matrix dimension (3 * num_nodes_in_component) is at most this use a dense
   * eigensolver; larger ones use the sparse iterative solver. Small problems are both faster and
   * more reliable dense, and the iterative solver cannot return the 4 eigenvalues needed for the
   * gap unless the dimension exceeds 4.
   */
  int dense_max_dim = 60;
};

/**
 * Result of a rotation synchronization.
 */
struct RotationSyncResult {
  /** Estimated rotation per node (local -> world). Identity for nodes with no surviving edge. */
  std::vector<Eigen::Matrix3d> rotations;
  /** False for a node that had no surviving edge, so its rotation was not estimated. */
  std::vector<bool> valid;
  /**
   * False for a node whose rounding disagreed with its component's global orientation, which
   * indicates that node's relative rotations are inconsistent with the rest of the component.
   */
  std::vector<bool> reliable;
  /** Connected-component id per node over the surviving edges; -1 for a node with no edge. */
  std::vector<int> component;
  /**
   * Per component (indexed by component id), the gap between the 3rd and 4th largest eigenvalues
   * of the normalized synchronization matrix. A small gap means the top-3 eigenspace -- and hence
   * the recovered rotations -- is poorly determined. -1 when the component is too small for a 4th
   * eigenvalue to exist.
   */
  std::vector<double> spectral_gap;
  /** Number of connected components over the surviving edges. */
  int num_components = 0;
};

/**
 * Noise bound on the residual ||R_p x_j - R_q y_j|| when x_j and y_j are TIM vectors, given the
 * per-point noise bounds of the two scans.
 *
 * A TIM is a difference of two measured points, so its own bound is the sum of its two endpoints'
 * bounds -- 2 * delta for a scan whose points share a bound delta. Both sides of the residual are
 * TIMs from real scans here, and both scans' rotations are free, so the two contributions add:
 *
 *     sigma_pq = 2 * delta_p + 2 * delta_q
 *
 * Note this is the SYMMETRIC composition, which is the right one when every rotation is being
 * estimated at once. It is deliberately not what the pairwise pipeline does: registration.cc:782
 * uses 2 * delta because TEASER's measurement model treats the source cloud as exact and attributes
 * all noise to the destination. Reusing that factor here would make the bound too tight by 2x when
 * both scans are equally noisy.
 *
 * @param delta_p [in] per-point noise bound of scan p
 * @param delta_q [in] per-point noise bound of scan q
 * @return the noise bound on a TIM-to-TIM residual between the two scans
 */
double timResidualNoiseBound(double delta_p, double delta_q);

/**
 * Jointly estimate every node's rotation by spectral rotation synchronization.
 *
 * Solves the aggregated weighted least-squares problem over all nodes at once
 *
 *     argmin_{R_i in SO(3)}  sum_{(p,q) in E} sum_j w_j || R_p x_j - R_q y_j ||^2
 *
 * where x_j / y_j are edge (p,q)'s TIM vectors p_tims / q_tims -- see RotationSyncEdge for why they
 * must be TIMs and not raw points.
 *
 * via the standard spectral relaxation: reduce each edge to its relative-rotation estimate
 * R_q^T R_p, assemble those blocks into a symmetric 3n-by-3n matrix, take its top-3 eigenspace,
 * and round each 3-by-3 block back onto SO(3).
 *
 * This is the multi-node analogue of teaser::utils::svdRot and is intended for the same slot in a
 * GNC loop: it consumes fixed line-process weights and returns rotations, leaving the caller to
 * recompute residuals and update the weights.
 *
 * Each connected component has its own arbitrary global rotation (gauge); it is fixed here so that
 * the lowest-indexed node of every component comes out as exactly the identity.
 *
 * @param num_nodes [in] number of nodes; edges must reference 0..num_nodes-1
 * @param edges [in] the synchronization graph edges with their weighted correspondences
 * @param params [in] tuning knobs
 * @param initial [in] optional per-node fallback rotations (must be sized num_nodes when given),
 *        used for nodes that end up with no surviving edge. Lets a GNC iteration keep a node's
 *        previous estimate instead of snapping it back to the identity when all of that node's
 *        correspondences are rejected.
 * @param node_noise_bounds [in] optional per-scan PER-POINT noise bounds (must be sized num_nodes
 *        when given; the TIM composition is what turns a per-point bound into a per-edge one -- see
 *        timResidualNoiseBound). Each edge is then weighted by 1 / sigma_pq^2, so a noisy scan's
 *        edges pull less on their neighbors. Uniform bounds are a no-op: a common factor on every
 *        edge weight cancels out of the normalized matrix. An edge is dropped if its composed
 *        sigma_pq is not positive (both endpoints declared exactly noiseless).
 * @return the per-node rotations plus validity, reliability, and conditioning diagnostics
 */
RotationSyncResult synchronizeRotations(int num_nodes, const std::vector<RotationSyncEdge>& edges,
                                        const RotationSyncParams& params = RotationSyncParams(),
                                        const std::vector<Eigen::Matrix3d>* initial = nullptr,
                                        const std::vector<double>* node_noise_bounds = nullptr);

} // namespace teaser
