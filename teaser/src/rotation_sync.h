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
  /** World "up" direction that the upright prior aligns gravity to. Normalized internally. */
  Eigen::Vector3d world_up = Eigen::Vector3d::UnitZ();
  /**
   * Strength of the optional VIRTUAL-NODE upright prior. 0 (the default) leaves it off.
   *
   * When positive, and per-scan gravity is supplied, each component is augmented with a fictitious
   * "world" node joined to every scan that has a reading, with the rank-one block
   * eta * g_i * up^T. Because that node touches everything it is a hub: it collapses the graph
   * diameter and sharply improves conditioning (measured: algebraic connectivity up ~240x on a
   * 60-node chain).
   *
   * The block is rank one, encoding only the two degrees of freedom gravity actually fixes. That
   * means the ground truth is no longer literally an eigenvector of the augmented matrix -- a
   * rotation block satisfies (R_q^T R_p) rho_p = rho_q, a rank-deficient one cannot. In practice
   * this costs nothing when the readings agree with the edges: the truth still saturates every term
   * of the objective, so recovery stays exact (measured below 1e-5 degrees of relative-rotation
   * error for eta up to 10 on exact, gravity-consistent data). When gravity DISAGREES with the edge
   * measurements the prior pulls the solution toward gravity in proportion to eta, which is the
   * intended behaviour of a prior rather than an artifact -- size eta by how much you trust the
   * readings relative to the correspondences.
   *
   * \attention It does NOT repair an upside-down scan. Gravity constrains pitch and roll and says
   * nothing about yaw, while a 180-degree flip is a tilt composed with a 180-degree yaw; tested
   * over eta from 0.05 to 10, the flip always survived. Use `tilt_error` to DETECT flipped scans
   * and re-estimate them from their neighbours; this prior cannot undo one.
   */
  double upright_prior_eta = 0.0;
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
  /**
   * Per component, the four largest eigenvalues (lambda_1 >= ... >= lambda_4, descending) of the
   * normalized synchronization matrix D^-1/2 B D^-1/2. NaN when the eigendecomposition was not run
   * or did not converge far enough. Note eigenvalues can legitimately be negative, so NaN -- not a
   * negative value -- is the "unavailable" marker.
   *
   * These measure something DIFFERENT from spectral_gap, and both are worth checking:
   *
   * - lambda_1..lambda_3 are all exactly 1 when the measured relative rotations are globally
   *   consistent (every cycle closes), because then the truth spans the leading eigenspace. So
   *   1 - lambda_3 is a normalized measure of how INCONSISTENT the data is, independent of how well
   *   connected the graph is. As a rough reading, a typical per-edge rotation inconsistency of
   *   angle eps gives 1 - lambda_3 ~ eps^2 / 3.
   * - spectral_gap (lambda_3 - lambda_4) instead measures IDENTIFIABILITY: whether the top-3
   *   eigenspace is separated enough for the relaxation to be tight and the rounding stable.
   *
   * A graph can be consistent but under-determined (long chain: 1 - lambda_3 tiny, gap tiny), or
   * well connected but inconsistent (one bad edge: gap fine, 1 - lambda_3 large). Checking only the
   * gap misses the second case.
   *
   * lambda_1 is 1 whenever the component is connected, so use lambda_3 (or the sum over the top
   * three) rather than lambda_1 as the consistency indicator.
   */
  std::vector<Eigen::Vector4d> top_eigenvalues;
  /**
   * Per component, the normalized algebraic connectivity (Fiedler value) mu_2 of the n-by-n SCAN
   * graph -- the second smallest eigenvalue of I - D^-1/2 A D^-1/2, where A_pq = c_pq. NaN for a
   * component with fewer than two nodes.
   *
   * Like spectral_gap, this describes the graph ACTUALLY SOLVED, so it includes the virtual world
   * node when RotationSyncParams::upright_prior_eta > 0 -- which is exactly how the prior's
   * conditioning benefit shows up here.
   *
   * This is PURE TOPOLOGY: it depends only on which scans are connected and their edge confidences,
   * never on the measured rotations. It is exposed because on exact (globally consistent) data it
   * equals the spectral gap identically,
   *
   *     lambda_3 - lambda_4  ==  mu_2,
   *
   * which follows from B_n = G (A_n kron I_3) G^T -- the spectrum of B_n is the spectrum of the
   * scalar normalized adjacency A_n with every eigenvalue tripled, so lambda_4 = alpha_2 = 1 - mu_2.
   *
   * Comparing the two therefore SEPARATES THE TWO CAUSES of a small gap:
   *
   * - gap ~= mu_2, both small  ->  the GRAPH is the problem. It is long and thin (a path scales as
   *   O(1/n^2)) or nearly split by a weak bridge. No amount of re-solving helps; add overlap between
   *   the weakly joined clusters, raise the confidence of a trusted bridge edge, or attach an
   *   absolute prior. mu_2 is cheap to evaluate and to optimize BEFORE running the solve.
   * - mu_2 healthy but gap << mu_2  ->  the DATA is the problem. The topology could support a
   *   well-determined answer, but inconsistent relative rotations are dragging lambda_3 below 1.
   *   Adding edges will not help; look at 1 - lambda_3 (top_eigenvalues) and at which edges carry
   *   the residual.
   *
   * Note mu_2 also measures something 1 - lambda_3 cannot: on a tree or chain there are no cycles,
   * so ANY set of relative rotations is consistent and 1 - lambda_3 is identically 0 however noisy
   * the data is. For chain-like graphs mu_2 and the gap are the only informative signals.
   */
  std::vector<double> algebraic_connectivity;
  /** Number of connected components over the surviving edges. */
  int num_components = 0;
  /**
   * Per node, the residual tilt in RADIANS: the angle between R_i * g_i and world up, i.e. how far
   * that scan's measured gravity ends up from vertical. NaN when no gravity was supplied for it.
   *
   * This is the payoff of the upright gauge and the intended **flip detector**. Without an absolute
   * reference the gauge is arbitrary and per-node tilt is meaningless -- four perfectly correct
   * nodes measured 118 degrees of "tilt" purely because the world frame was unpinned. Once the gauge
   * is upright, a scan whose tilt is an outlier near pi is upside down.
   *
   * Note that detecting a flip is not fixing it: gravity leaves yaw free, so repairing a flipped
   * scan means re-estimating it from its already-posed neighbours.
   */
  std::vector<double> tilt_error;
  /**
   * Per component, true if the gauge was fixed by aligning gravity to world up, false if it fell
   * back to pinning the component's lowest-indexed node to the identity (no usable gravity).
   *
   * Callers that re-gauge downstream must not undo an upright gauge -- see how
   * teaser::alignMultiScanWithGraph makes its own re-gauge conditional on this.
   */
  std::vector<bool> gauge_upright;
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
 * @param node_gravity [in] optional per-scan gravity, expressed in each scan's OWN local frame and
 *        sized num_nodes. Supplying it switches the gauge from "anchor is identity" to "gravity
 *        points up", which is a pure gauge change and so leaves every relative rotation untouched,
 *        and it populates `tilt_error`. A zero vector marks a scan with no reading, which is
 *        skipped. It additionally enables the virtual-node prior when
 *        RotationSyncParams::upright_prior_eta > 0.
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
                                        const std::vector<double>* node_noise_bounds = nullptr,
                                        const std::vector<Eigen::Vector3d>* node_gravity = nullptr);

} // namespace teaser
