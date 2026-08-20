/**
 * Copyright 2020, Massachusetts Institute of Technology,
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

#include "teaser/geometry.h"
#include "teaser/graph.h"
#include "teaser/registration.h"

namespace teaser {

/**
 * A weighted undirected edge between two graph vertices (e.g. two scans).
 */
struct WeightedEdge {
  int u;
  int v;
  double weight;
};

/**
 * Kruskal's algorithm: compute a maximum- (default) or minimum-weight spanning tree.
 *
 * Vertices are assumed to be labeled 0..num_vertices-1. Edges referencing out-of-range or
 * negative vertices are ignored. Self-loops and duplicate/parallel edges are handled naturally
 * (the first-encountered edge in sorted order wins).
 *
 * @param num_vertices [in] number of vertices (union-find size)
 * @param edges [in] the weighted undirected edges
 * @param maximum [in] if true (default) select maximum-weight edges, otherwise minimum-weight
 * @return the selected edges. For a connected graph this is a spanning tree with
 *         (num_vertices - 1) edges; for a disconnected graph it is a spanning forest with fewer
 *         edges (so `result.size() == num_vertices - 1` iff the graph is connected). Edges are
 *         returned in selection order (by descending weight when maximum, ascending otherwise).
 */
std::vector<WeightedEdge> kruskalSpanningTree(int num_vertices,
                                              const std::vector<WeightedEdge>& edges,
                                              bool maximum = true);

/**
 * Topological sort (Kahn's algorithm) over a directed acyclic graph.
 *
 * Each edge (u, v) means u is a parent of v, i.e. u must come before v. Vertices are labeled
 * 0..num_vertices-1. The returned ordering lists every vertex only after all of its parents have
 * appeared. When several vertices are simultaneously ready (all parents already placed), the
 * smallest index is emitted first, so the ordering is deterministic. Edges referencing
 * out-of-range or negative vertices are ignored.
 *
 * @param num_vertices [in] number of vertices
 * @param edges [in] directed parent->child edges
 * @return a topological ordering containing all num_vertices vertices, or an empty vector if the
 *         graph contains a cycle (in which case no valid ordering exists).
 */
std::vector<int> topologicalSort(int num_vertices,
                                 const std::vector<std::pair<int, int>>& edges);

/**
 * One overlapping neighbor edge for the node whose global pose is being optimized.
 *
 * The neighbor's global pose (R_i, t_i) is held fixed and maps neighbor-local coordinates into
 * the world frame (i.e. world = R_i * local + t_i). Correspondences are pre-matched column-wise:
 * src.col(k) (a point in the neighbor's local frame) corresponds to dst.col(k) (a point in the
 * frame of the node being optimized).
 */
struct NeighborEdge {
  /** Neighbor's fixed global rotation (local -> world). */
  Eigen::Matrix3d R_i;
  /** Neighbor's fixed global translation (local -> world). */
  Eigen::Vector3d t_i;
  /** Neighbor-local source points, 3-by-K_i. */
  Eigen::Matrix<double, 3, Eigen::Dynamic> src;
  /** Node-local target points, 3-by-K_i (column i <-> src column i). */
  Eigen::Matrix<double, 3, Eigen::Dynamic> dst;
  /** Weight applied to every correspondence of this edge in the aggregated TLS (default 1). */
  double weight = 1.0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Estimate the global pose of a single node from correspondences to already-posed neighbors.
 *
 * This is a thin extension of the pairwise TEASER solver to the multiview setting, as described in
 * multiview.md. It optimizes the pose of one node (call it B) while holding all of its neighbors'
 * poses fixed. Scale is assumed to be 1.
 *
 * The estimator does not perform any graph-wide optimization: the caller is responsible for
 * choosing which node to optimize and, if desired, sweeping over nodes. It also does not run
 * feature matching -- correspondences are provided directly as point pairs.
 */
class MultiviewSolver {
public:
  MultiviewSolver() = delete;

  explicit MultiviewSolver(const RobustRegistrationSolver::Params& params) : params_(params) {}

  /**
   * Estimate node B's global pose (R_B, t_B) given its edges to fixed neighbors.
   *
   * Each edge contributes correspondences between a fixed neighbor and node B. All edges are
   * expressed in the world frame and stacked into a single robust registration problem; the
   * resulting world->B transform is inverted to recover B's global (B->world) pose.
   *
   * @param edges [in] the set of neighbor edges for node B
   * @return a RegistrationSolution with rotation = R_B, translation = t_B, scale = 1. The `valid`
   *         flag is false if the inputs are degenerate (fewer than 3 total correspondences).
   */
  RegistrationSolution solveNodePose(const std::vector<NeighborEdge>& edges);

private:
  RobustRegistrationSolver::Params params_;
};

/**
 * A rigid global pose (local -> world): world = R * local + t.
 */
struct Pose {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Result of a multi-scan alignment: one global pose per input scan.
 */
struct MultiScanResult {
  /** Global pose per node. The anchor of each connected component is the identity. */
  std::vector<Pose> poses;
  /** Per-node validity (false if that node's alignment failed or it was unreachable). */
  std::vector<bool> valid;
  /** Connected-component id per node (nodes in the same component share a gauge). */
  std::vector<int> component;
  /**
   * Per-edge fit quality, keyed by (min(i,j), max(i,j)): the mean world-frame residual over that
   * edge's inlier correspondences (those whose post-alignment residual is within the noise bound).
   * The value is -1 for an edge that retained no inliers, which is what a rejected edge looks like.
   *
   * Which edges appear depends on params.multiview_method. Under SPECTRAL_SYNC every edge that
   * survived preprocessing gets an entry -- the joint solve uses them all, so there is no tree-edge
   * / loop-closure distinction. Under DAG_PROPAGATION only edges actually used to pose a node
   * appear: one per incoming edge of each aligned node, so an anchor's outgoing side and anything
   * pruned by the spanning tree are absent. Either way, edges with no correspondences, too few to
   * survive the max-clique prune, or whose endpoints could not be aligned have no entry.
   */
  std::map<std::pair<int, int>, double> edge_residual;
  /** Number of connected components found in the adjacency graph. */
  int num_components = 0;
};

/**
 * Optional gravity ("upright") prior for the multi-scan solvers.
 *
 * The rotation stage only ever sees RELATIVE rotations, so the recovered frame is arbitrary and a
 * scan that comes out upside down -- a 180-degree rotation, which is a perfectly proper rotation --
 * is invisible. Supplying gravity adds the missing absolute reference. Default-constructed, this
 * struct disables the prior entirely and the solvers behave exactly as before.
 *
 * Two mechanisms, with very different cost:
 *
 * 1. **Upright gauge** -- applied automatically whenever `gravity` is supplied. The solution is
 *    determined only up to a global rotation per component, so that rotation is CHOSEN to stand the
 *    map upright instead of pinning an arbitrary anchor to the identity. Being a gauge change it is
 *    exactly zero-bias: every relative rotation is untouched. Its real value is that per-node tilt
 *    then becomes meaningful, which turns it into a pose-free detector for upside-down scans.
 * 2. **Virtual node** -- opt-in via `virtual_node_eta > 0`. Gravity is folded into the
 *    eigenproblem as a fictitious world node joined to every scan. That node is a hub, so it
 *    collapses the graph diameter and sharply improves conditioning (algebraic connectivity rose
 *    ~240x on a 60-node chain in testing). It is off by default simply because it needs a strength
 *    chosen against how much the gravity readings are trusted: with readings that agree with the
 *    correspondences it costs no accuracy at all, but where they disagree it pulls the solution
 *    toward gravity in proportion to eta. See RotationSyncParams::upright_prior_eta.
 *
 * \attention Neither mechanism REPAIRS an upside-down scan. Gravity pins pitch and roll and says
 * nothing about yaw, whereas a 180-degree flip is a tilt composed with a 180-degree yaw. Use the
 * reported tilt to find the bad scan, then re-estimate it from its already-posed neighbours.
 */
struct UprightPrior {
  /**
   * Gravity in each scan's OWN local frame, sized clouds.size(). Empty (the default) disables the
   * prior. A zero vector marks a scan with no reading, which is skipped -- so partial coverage is
   * fine. Magnitude is irrelevant (normalized internally); for scans from a levelled mount, passing
   * the same vector for every scan is correct and sufficient.
   */
  std::vector<Eigen::Vector3d> gravity;
  /** The world "up" direction that `gravity` should map onto. Normalized internally. */
  Eigen::Vector3d world_up = Eigen::Vector3d::UnitZ();
  /**
   * Strength of the virtual-node prior. 0 (default) means gauge alignment only. Positive values
   * additionally improve conditioning; they cost nothing on gravity-consistent data, and where the
   * readings disagree with the correspondences they weigh gravity against them in proportion to
   * eta. Start around 0.1-1 relative to typical edge confidences.
   */
  double virtual_node_eta = 0.0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Align a set of overlapping scans into a common frame per connected component.
 *
 * Shared pipeline: split the adjacency graph into connected components (a warning is emitted if
 * there is more than one), and within each component pick the node with the largest total edge
 * weight as the anchor (identity pose). `params.multiview_method` then selects the optimization:
 *
 * MULTIVIEW_METHOD::SPECTRAL_SYNC (default) solves ALL poses at once -- rotation synchronization
 * then translation synchronization, both driven by a graph-level GNC-TLS loop over the
 * max-clique-pruned correspondences. Every adjacency edge participates: no spanning-tree prune,
 * because a tree is exactly determined and synchronizing over one would merely reproduce
 * propagation. Keeping every edge is what lets a loop closure spread its error around the cycle
 * instead of dumping it on one edge, and lets the graph out-vote an edge that is coherently wrong.
 *
 * MULTIVIEW_METHOD::DAG_PROPAGATION prunes to the maximum spanning tree and then sweeps outward
 * from the anchor, solving one pose at a time against its already-posed neighbors via
 * MultiviewSolver::solveNodePose. Each pose is frozen on arrival, so error accumulates along paths;
 * cheaper, and each node's estimate traces to a single local solve.
 *
 * The method therefore changes both the edge set consumed (full graph vs. spanning tree) and the
 * number of entries in MultiScanResult::edge_residual.
 *
 * `edge_weights` is documented as a spanning-tree weight independent of the correspondences, so it
 * is used ONLY to score anchors -- it is deliberately not reused as a per-edge cost weight. Use
 * alignMultiScanWithGraph if you want caller-supplied weights to influence the solve.
 *
 * Correspondences are the raw putative matches (outliers included); the per-edge robust solve
 * performs its own inlier selection.
 *
 * @param clouds [in] clouds[i] is scan i (points in scan i's local frame)
 * @param adjacency [in] which scans overlap; vertices must be 0..clouds.size()-1
 * @param edge_weights [in] MST edge weight keyed by (min(i,j), max(i,j)); adjacency edges absent
 *        from the map are treated as weight 0
 * @param correspondences [in] keyed by (min(i,j), max(i,j)); each pair (a,b) is
 *        (index into cloud[min], index into cloud[max])
 * @param params [in] parameters forwarded to the per-node robust solve
 * @param upright [in] optional gravity prior; default-constructed disables it. When supplied, each
 *        component's gauge is chosen to stand the map upright rather than pinning its anchor to the
 *        identity -- so poses[anchor] is no longer the identity rotation (its translation is still
 *        zero). See UprightPrior.
 * @return per-node global poses, validity, and component labeling
 */
MultiScanResult alignMultiScan(
    const std::vector<teaser::PointCloud>& clouds, const teaser::Graph& adjacency,
    const std::map<std::pair<int, int>, double>& edge_weights,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const RobustRegistrationSolver::Params& params, const UprightPrior& upright = UprightPrior());

/**
 * Align a set of overlapping scans given a caller-provided graph (MST step skipped).
 *
 * Identical to alignMultiScan except that the graph is supplied directly, so no spanning-tree prune
 * happens under either method -- every supplied edge is used as given. `graph_edges` are undirected
 * `(i, j)` pairs (duplicates ignored); the graph may or may not be a tree. Connected components are
 * derived from the edges, and each component's anchor is the highest-degree node (ties -> smallest
 * index). `params.multiview_method` selects the optimization, as for alignMultiScan. The returned
 * gauge (per-component identity anchor) matches alignMultiScan.
 *
 * Optionally the caller can pass `order`, a preference order over the scans (root-most first). What
 * it controls depends on the method:
 *
 * - DAG_PROPAGATION: it fixes each component's anchor (the earliest listed node) AND the sweep
 *   sequence, since poses propagate in that order. A node whose neighbors are all listed later is
 *   still aligned once one of them is posed (BFS fallback), so an inconsistent order never leaves
 *   nodes unaligned.
 * - SPECTRAL_SYNC: all poses are solved simultaneously, so there is no sequence left to control and
 *   it does exactly one thing -- select the anchor. Since the anchor only fixes the gauge, an order
 *   inconsistent with the graph is harmless.
 *
 * Nodes absent from a non-empty `order` rank after all listed ones under both methods.
 *
 * @param clouds [in] clouds[i] is scan i (points in scan i's local frame)
 * @param graph_edges [in] undirected graph edges over vertices 0..clouds.size()-1 (tree or not)
 * @param correspondences [in] keyed by (min(i,j), max(i,j)); each pair (a,b) is
 *        (index into cloud[min], index into cloud[max])
 * Optionally the caller can pass `edge_weights`, one weight per entry of `graph_edges`. Edge `i`'s
 * weight scales every one of its correspondences in the aggregated rotation and translation TLS
 * costs. Empty = all edges weight 1. NOTE: the weight scales the cost among the max-clique inliers;
 * it does not influence max-clique inlier selection or the scale-consistency prune.
 *
 * @param params [in] parameters forwarded to the per-node robust solve
 * @param order [in] optional topological order of node indices (root-most first); empty = auto
 * @param edge_weights [in] optional per-edge weights aligned with `graph_edges`; empty = unweighted
 * @return per-node global poses, validity, and component labeling
 */
MultiScanResult alignMultiScanWithGraph(
    const std::vector<teaser::PointCloud>& clouds,
    const std::vector<std::pair<int, int>>& graph_edges,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const RobustRegistrationSolver::Params& params, const std::vector<int>& order = {},
    const std::vector<double>& edge_weights = {},
    const UprightPrior& upright = UprightPrior());

} // namespace teaser
