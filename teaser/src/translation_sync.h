/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <map>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace teaser {

/**
 * One edge (p, q) of the translation synchronization graph.
 *
 * Unlike the rotation stage these are RAW points, not TIMs. Differencing points into TIMs is what
 * cancels translation, which is exactly what must NOT happen here -- translation is the unknown.
 * Correspondences are pre-matched column-wise: p_pts.col(j) and q_pts.col(j) are the same world
 * point seen in the two scans' local frames, so at the optimum
 * R_p * p_pts.col(j) + t_p == R_q * q_pts.col(j) + t_q.
 *
 * No recentering is applied or wanted. Shifting cloud p by o_p moves this edge's target by
 * (R_p o_p - R_q o_q), and the corresponding undo t_p -= R_p o_p cancels it exactly, so the
 * anchored answer is unchanged; it is also ill-defined, since the weights are per-edge and a point
 * may appear on several edges with different weights.
 */
struct TranslationSyncEdge {
  /** Index of the first endpoint. */
  int p = -1;
  /** Index of the second endpoint. */
  int q = -1;
  /** Raw points in p's local frame, 3-by-m. */
  Eigen::Matrix<double, 3, Eigen::Dynamic> p_pts;
  /** Raw points in q's local frame, 3-by-m (column j <-> p_pts column j). */
  Eigen::Matrix<double, 3, Eigen::Dynamic> q_pts;
  /** Per-correspondence weights, 1-by-m. The GNC line-process weights, as in the rotation stage. */
  Eigen::Matrix<double, 1, Eigen::Dynamic> w;
  /**
   * Whole-edge confidence. Multiplied by 1 / sigma_pq^2 when per-scan noise bounds are supplied,
   * and by the edge's total correspondence weight, to form the Laplacian edge weight.
   */
  double confidence = 1.0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Tuning knobs for synchronizeTranslations.
 */
struct TranslationSyncParams {
  /** Edges whose total correspondence weight (sum_j w_j) falls below this are dropped. */
  double min_edge_mass = 1e-9;
  /** Compute the optional conditioning diagnostics. Off by default: they cost an extra solve. */
  bool compute_diagnostics = false;
  /** Components larger than this skip the Fiedler value, which needs a dense eigendecomposition. */
  int fiedler_max_nodes = 200;
};

/**
 * Result of a translation synchronization.
 */
struct TranslationSyncResult {
  /** Estimated translation per node (local -> world). Zero for the anchor of each component. */
  std::vector<Eigen::Vector3d> translations;
  /** False for a node that had no surviving edge, so its translation was not estimated. */
  std::vector<bool> valid;
  /** Connected-component id per node over the surviving edges; -1 for a node with no edge. */
  std::vector<int> component;
  /** Number of connected components. Matches the rotation stage given the same surviving edges. */
  int num_components = 0;
  /**
   * Per component: the residual scale, sqrt(weighted residual sum / degrees of freedom). With
   * per-scan noise bounds supplied the weights carry 1/sigma^2, so this is dimensionless and should
   * land near 1 when the noise model is honest; well above 1 means the edges disagree by more than
   * the declared bounds. -1 when the component is a spanning tree, which has no redundancy to
   * estimate it from, or when diagnostics are off.
   */
  std::vector<double> residual_sigma;
  /**
   * Per component: lambda_2 of the Laplacian (the Fiedler value). Small means the component is
   * nearly disconnected and relative translations across the bottleneck are poorly determined --
   * the translation analogue of the rotation stage's spectral gap. -1 when not computed.
   */
  std::vector<double> fiedler_value;
  /**
   * For each requested pair, the 1-sigma uncertainty of the relative translation t_p - t_q,
   * residual_sigma * sqrt(effective resistance). -1 if the pair spans two components or its
   * component has no residual_sigma. Empty unless pairs_of_interest was supplied.
   */
  std::map<std::pair<int, int>, double> pair_sigma;
};

/**
 * Noise bound on the residual ||(R_p p_j + t_p) - (R_q q_j + t_q)|| given the two scans' per-point
 * noise bounds.
 *
 *     sigma_pq = delta_p + delta_q
 *
 * Note there is NO factor of 2 here, unlike timResidualNoiseBound in the rotation stage. That
 * factor comes from a TIM being a difference of two measured points; translation consumes raw
 * points, so each scan contributes its per-point bound once. The pairwise pipeline draws the same
 * distinction -- registration.cc uses beta = noise_bound for translation but 2 * noise_bound for
 * the TIM-based tests.
 *
 * @param delta_p [in] per-point noise bound of scan p
 * @param delta_q [in] per-point noise bound of scan q
 * @return the noise bound on a point-to-point residual between the two scans
 */
double pointResidualNoiseBound(double delta_p, double delta_q);

/**
 * Jointly estimate every node's translation, given the rotations.
 *
 * Solves the aggregated weighted least-squares problem over all nodes at once
 *
 *     argmin_{t_i}  sum_{(p,q) in E} sum_j w_j || (R_p p_j + t_p) - (R_q q_j + t_q) ||^2
 *
 * With the rotations fixed this is linear, and its normal equations are a weighted graph Laplacian
 * system L T = Bm with three right-hand sides -- one sparse factorization for all of x, y and z.
 * Each edge first collapses to a single weighted-centroid target, which is EXACT rather than an
 * approximation: the inner sum depends on t_p and t_q only through their difference.
 *
 * NOTE this is plain weighted least squares. It is the correct inner step of a GNC loop over
 * translation (where w carries the robustness), but run once on raw putative matches nothing
 * rejects outliers. A GNC caller must also keep p_pts/q_pts to rebuild per-correspondence
 * residuals for the weight update -- the per-edge reduction throws those away.
 *
 * Gauge: each component's lowest-indexed node is pinned to zero translation, the same anchor
 * synchronizeRotations pins to the identity, so the two results compose directly.
 *
 * @param num_nodes [in] number of nodes; edges must reference 0..num_nodes-1
 * @param edges [in] the synchronization graph edges with their weighted correspondences
 * @param rotations [in] per-node rotations from the rotation stage; must be sized num_nodes
 * @param params [in] tuning knobs
 * @param initial [in] optional per-node fallback translations (sized num_nodes when given), used
 *        for nodes with no surviving edge
 * @param node_noise_bounds [in] optional per-scan per-point noise bounds (sized num_nodes when
 *        given). Each edge is then weighted by 1 / sigma_pq^2. Uniform bounds rescale every edge
 *        weight by a common factor, which cancels, so they leave the answer unchanged.
 * @param pairs_of_interest [in] optional node pairs to report relative-translation uncertainty for;
 *        requires compute_diagnostics
 * @return the per-node translations plus validity and conditioning diagnostics
 */
TranslationSyncResult synchronizeTranslations(
    int num_nodes, const std::vector<TranslationSyncEdge>& edges,
    const std::vector<Eigen::Matrix3d>& rotations,
    const TranslationSyncParams& params = TranslationSyncParams(),
    const std::vector<Eigen::Vector3d>* initial = nullptr,
    const std::vector<double>* node_noise_bounds = nullptr,
    const std::vector<std::pair<int, int>>* pairs_of_interest = nullptr);

} // namespace teaser
