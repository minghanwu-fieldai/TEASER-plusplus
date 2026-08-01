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
   * Mean world-frame residual of the edge(s) connecting each node to its already-posed parent(s),
   * averaged over the inlier correspondences (those whose post-alignment residual is within the
   * noise bound) across all of the node's parent edges. This is a per-node fit-quality metric; with
   * tree-based paths each node has a single parent, while the graph path may aggregate several. It
   * is negative (-1) for component anchors (which have no parent), for nodes whose alignment failed,
   * and when no inliers remain.
   */
  std::vector<double> residual_to_parent;
  /** Number of connected components found in the adjacency graph. */
  int num_components = 0;
};

/**
 * Align a set of overlapping scans into a common frame per connected component.
 *
 * Pipeline: build a maximum spanning tree over the adjacency graph using the caller-supplied
 * edge weights; split into connected components (a warning is emitted if there is more than one);
 * within each component pick the node with the largest total edge weight as the anchor (identity
 * pose) and propagate poses outward along the tree in topological order, aligning each child to
 * its single tree-parent via teaser::MultiviewSolver::solveNodePose.
 *
 * The MST edge weight is provided by the caller and is independent of the correspondences, so the
 * tree can be chosen from any proxy (overlap, proximity, keypoint count, ...). Because propagation
 * uses tree edges only, correspondences are consulted for tree edges only -- the caller may supply
 * them for just those edges if the tree is known in advance.
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
 * @return per-node global poses, validity, and component labeling
 */
MultiScanResult alignMultiScan(
    const std::vector<teaser::PointCloud>& clouds, const teaser::Graph& adjacency,
    const std::map<std::pair<int, int>, double>& edge_weights,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const RobustRegistrationSolver::Params& params);

/**
 * Align a set of overlapping scans given a caller-provided graph (MST step skipped).
 *
 * Identical to alignMultiScan except that the graph is supplied directly instead of being computed
 * via a maximum spanning tree. The graph may or may not be a tree: `graph_edges` are undirected
 * `(i, j)` pairs (duplicates ignored). Connected components are derived from the edges, and each
 * component's anchor (its root) is the highest-degree node (ties -> smallest index). The root is
 * used to run a BFS that assigns a discovery order; every edge is then oriented from the
 * earlier-discovered endpoint (parent) to the later one (child). Because that orientation follows a
 * total order it is always acyclic, so loop-closure edges are kept rather than pruned: a node with
 * several in-edges is aligned to ALL of its already-posed parents at once (the multi-edge
 * aggregation from multiview.md). The returned gauge (per-component identity anchor) matches
 * alignMultiScan.
 *
 * @param clouds [in] clouds[i] is scan i (points in scan i's local frame)
 * @param graph_edges [in] undirected graph edges over vertices 0..clouds.size()-1 (tree or not)
 * @param correspondences [in] keyed by (min(i,j), max(i,j)); each pair (a,b) is
 *        (index into cloud[min], index into cloud[max])
 * @param params [in] parameters forwarded to the per-node robust solve
 * @return per-node global poses, validity, and component labeling
 */
MultiScanResult alignMultiScanWithGraph(
    const std::vector<teaser::PointCloud>& clouds,
    const std::vector<std::pair<int, int>>& graph_edges,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const RobustRegistrationSolver::Params& params);

} // namespace teaser
