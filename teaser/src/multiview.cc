/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "teaser/multiview.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <utility>

#include "teaser/registration.h"

#include "pose_refine.h"
#include "rotation_sync.h"
#include "translation_sync.h"

std::vector<teaser::WeightedEdge>
teaser::kruskalSpanningTree(int num_vertices, const std::vector<teaser::WeightedEdge>& edges,
                            bool maximum) {
  std::vector<teaser::WeightedEdge> tree;
  if (num_vertices <= 0) {
    return tree;
  }

  // Sort edges by weight: descending for a maximum spanning tree, ascending for a minimum one.
  std::vector<teaser::WeightedEdge> sorted = edges;
  std::sort(sorted.begin(), sorted.end(),
            [maximum](const teaser::WeightedEdge& a, const teaser::WeightedEdge& b) {
              return maximum ? (a.weight > b.weight) : (a.weight < b.weight);
            });

  // Union-find with path halving + union by rank.
  std::vector<int> parent(num_vertices);
  std::iota(parent.begin(), parent.end(), 0);
  std::vector<int> rank_(num_vertices, 0);
  auto find = [&parent](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]]; // path halving
      x = parent[x];
    }
    return x;
  };

  tree.reserve(static_cast<size_t>(num_vertices - 1));
  for (const auto& e : sorted) {
    // Skip edges that reference vertices outside [0, num_vertices).
    if (e.u < 0 || e.v < 0 || e.u >= num_vertices || e.v >= num_vertices) {
      continue;
    }
    int ru = find(e.u);
    int rv = find(e.v);
    if (ru == rv) {
      continue; // would form a cycle (also covers self-loops)
    }
    // Union by rank.
    if (rank_[ru] < rank_[rv]) {
      std::swap(ru, rv);
    }
    parent[rv] = ru;
    if (rank_[ru] == rank_[rv]) {
      ++rank_[ru];
    }
    tree.push_back(e);
    if (static_cast<int>(tree.size()) == num_vertices - 1) {
      break; // spanning tree complete
    }
  }
  return tree;
}

std::vector<int>
teaser::topologicalSort(int num_vertices, const std::vector<std::pair<int, int>>& edges) {
  std::vector<int> order;
  if (num_vertices <= 0) {
    return order;
  }

  // Build child adjacency and in-degrees (number of parents) per vertex.
  std::vector<std::vector<int>> children(num_vertices);
  std::vector<int> in_degree(num_vertices, 0);
  for (const auto& e : edges) {
    const int u = e.first;  // parent
    const int v = e.second; // child
    if (u < 0 || v < 0 || u >= num_vertices || v >= num_vertices) {
      continue; // ignore out-of-range edges
    }
    children[u].push_back(v);
    ++in_degree[v];
  }

  // Kahn's algorithm. Min-heap on vertex id => smallest ready index first (deterministic).
  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (int i = 0; i < num_vertices; ++i) {
    if (in_degree[i] == 0) {
      ready.push(i);
    }
  }

  order.reserve(num_vertices);
  while (!ready.empty()) {
    const int u = ready.top();
    ready.pop();
    order.push_back(u);
    for (const int v : children[u]) {
      if (--in_degree[v] == 0) {
        ready.push(v);
      }
    }
  }

  // If not all vertices were emitted, the graph has a cycle => no valid ordering.
  if (static_cast<int>(order.size()) != num_vertices) {
    order.clear();
  }
  return order;
}

teaser::RegistrationSolution
teaser::MultiviewSolver::solveNodePose(const std::vector<teaser::NeighborEdge>& edges) {
  // Count total correspondences across all edges.
  Eigen::Index total = 0;
  for (const auto& edge : edges) {
    assert(edge.src.cols() == edge.dst.cols());
    if (edge.src.cols() != edge.dst.cols()) {
      // Inconsistent edge: cannot use it.
      continue;
    }
    total += edge.src.cols();
  }

  // Rotation from TIMs needs at least 2 correspondences, and the scalar TLS estimator asserts on
  // the single-element case; require a small minimum.
  if (total < 3) {
    teaser::RegistrationSolution invalid;
    invalid.valid = false;
    invalid.scale = 1.0;
    invalid.rotation = Eigen::Matrix3d::Identity();
    invalid.translation = Eigen::Vector3d::Zero();
    return invalid;
  }

  // Stack every edge into a single (src, dst) problem, expressing each neighbor's source points in
  // the world frame: src_world = R_i * a^i + t_i, dst_local = b^i. Each edge's `weight` is
  // replicated across its correspondences so it scales every term of that edge in the aggregated
  // TLS.
  Eigen::Matrix<double, 3, Eigen::Dynamic> src_world(3, total);
  Eigen::Matrix<double, 3, Eigen::Dynamic> dst_local(3, total);
  std::vector<double> corr_weights(static_cast<size_t>(total), 1.0);
  bool weighted = false;
  Eigen::Index offset = 0;
  for (const auto& edge : edges) {
    const Eigen::Index k = edge.src.cols();
    if (k == 0 || edge.src.cols() != edge.dst.cols()) {
      continue;
    }
    src_world.middleCols(offset, k) = (edge.R_i * edge.src).colwise() + edge.t_i;
    dst_local.middleCols(offset, k) = edge.dst;
    std::fill_n(corr_weights.begin() + offset, k, edge.weight);
    if (edge.weight != 1.0) {
      weighted = true;
    }
    offset += k;
  }

  // Solve the robust registration with scale fixed to 1. With src = world and dst = B-local, the
  // solver returns the world->B transform, i.e. rotation = R_B^T, translation = -R_B^T * t_B.
  // Pass the per-correspondence weights only when some edge is non-default (keeps the unweighted
  // solve path byte-for-byte otherwise).
  teaser::RobustRegistrationSolver::Params p = params_;
  p.estimate_scaling = false;
  teaser::RobustRegistrationSolver solver(p);
  teaser::RegistrationSolution sol =
      weighted ? solver.solve(src_world, dst_local, corr_weights) : solver.solve(src_world, dst_local);

  // Invert the world->B transform to recover B's global (B->world) pose.
  teaser::RegistrationSolution out;
  out.valid = sol.valid;
  out.scale = 1.0;
  out.rotation = sol.rotation.transpose();           // R_B = (R_B^T)^T
  out.translation = -out.rotation * sol.translation; // t_B = -R_B * (-R_B^T t_B)
  return out;
}

namespace {

using Pts = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using RowVec = Eigen::Matrix<double, 1, Eigen::Dynamic>;

// A graph edge reduced to what the two synchronization stages consume: the max-clique-pruned raw
// correspondences (translation needs raw points) and their TIMs (rotation needs translation-free
// measurements).
struct PreparedEdge {
  int p = -1;
  int q = -1;
  double confidence = 1.0;
  Pts p_pts, q_pts;   // clique correspondences, each scan's local frame
  Pts p_tims, q_tims; // TIMs of those points

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ============================ MULTIVIEW_METHOD::DAG_PROPAGATION ============================

// Build a NeighborEdge aligning `child` (target/dst) to an already-posed `parent` (fixed source)
// over their shared correspondences. Returns false if no usable correspondence columns remain.
bool buildNeighborEdge(const std::vector<teaser::PointCloud>& clouds,
                       const std::vector<std::pair<int, int>>& corr, int parent, int child,
                       const teaser::Pose& parent_pose, teaser::NeighborEdge* edge) {
  const bool parent_is_min = (parent < child);
  edge->R_i = parent_pose.R;
  edge->t_i = parent_pose.t;
  edge->src.resize(3, static_cast<Eigen::Index>(corr.size()));
  edge->dst.resize(3, static_cast<Eigen::Index>(corr.size()));
  Eigen::Index col = 0;
  for (const auto& c : corr) {
    const int p_idx = parent_is_min ? c.first : c.second; // index into parent cloud
    const int b_idx = parent_is_min ? c.second : c.first; // index into child cloud
    if (p_idx < 0 || p_idx >= static_cast<int>(clouds[parent].size()) || b_idx < 0 ||
        b_idx >= static_cast<int>(clouds[child].size())) {
      continue; // skip malformed correspondence
    }
    const auto& pp = clouds[parent][p_idx];
    const auto& bp = clouds[child][b_idx];
    edge->src.col(col) << pp.x, pp.y, pp.z;
    edge->dst.col(col) << bp.x, bp.y, bp.z;
    ++col;
  }
  edge->src.conservativeResize(3, col);
  edge->dst.conservativeResize(3, col);
  return col > 0;
}

// Sequential propagation. Sweep nodes in reference order to a fixpoint, aligning each not-yet-
// settled node that has at least one already-posed neighbor to ALL such neighbors via
// MultiviewSolver::solveNodePose. A node whose neighbors are all posed later is picked up in a
// subsequent sweep (BFS fallback from the anchor), so an order inconsistent with the graph never
// leaves nodes unaligned. Anchors must already carry the identity pose and be marked in `is_anchor`.
void alignByPropagation(
    const std::vector<teaser::PointCloud>& clouds, const std::vector<std::vector<int>>& adj,
    const std::vector<int>& ref,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const teaser::RobustRegistrationSolver::Params& params,
    const std::map<std::pair<int, int>, double>& edge_weight, const std::vector<char>& is_anchor,
    teaser::MultiScanResult* result) {
  const int N = static_cast<int>(clouds.size());
  auto edge_key = [](int a, int b) { return std::make_pair(std::min(a, b), std::max(a, b)); };

  // Sweep order: nodes sorted by ref (the caller order, or the BFS discovery order).
  std::vector<int> proc(N);
  std::iota(proc.begin(), proc.end(), 0);
  std::stable_sort(proc.begin(), proc.end(), [&ref](int a, int b) { return ref[a] < ref[b]; });

  std::vector<char> settled(N, 0);
  for (int i = 0; i < N; ++i) {
    if (is_anchor[i]) {
      settled[i] = 1;
    }
  }

  teaser::MultiviewSolver solver(params);
  bool changed = true;
  while (changed) {
    changed = false;
    for (const int node : proc) {
      if (settled[node] || is_anchor[node]) {
        continue;
      }
      std::vector<teaser::NeighborEdge> node_edges;
      std::vector<int> used_parents;
      for (const int p : adj[node]) {
        if (!result->valid[p]) {
          continue; // neighbor not posed yet (or failed)
        }
        const auto it = correspondences.find(edge_key(p, node));
        if (it == correspondences.end() || it->second.empty()) {
          continue;
        }
        teaser::NeighborEdge edge;
        if (buildNeighborEdge(clouds, it->second, p, node, result->poses[p], &edge)) {
          const auto wit = edge_weight.find(edge_key(p, node));
          edge.weight = (wit != edge_weight.end()) ? wit->second : 1.0;
          node_edges.push_back(std::move(edge));
          used_parents.push_back(p);
        }
      }
      if (node_edges.empty()) {
        continue; // no posed neighbor yet -> defer to a later sweep
      }

      settled[node] = 1; // settled whether or not the solve succeeds
      changed = true;

      const teaser::RegistrationSolution sol = solver.solveNodePose(node_edges);
      result->poses[node].R = sol.rotation;
      result->poses[node].t = sol.translation;
      result->valid[node] = sol.valid;
      if (!sol.valid) {
        std::cerr << "[teaser::multiview] Warning: alignment failed for node " << node << ".\n";
        continue;
      }

      // Per-edge fit quality: mean world-frame residual over each edge's inlier correspondences
      // (those consistent with the recovered pose within the noise bound).
      const double inlier_thresh =
          2.0 * params.noise_bound * std::sqrt(std::max(0.0, params.cbar2));
      const teaser::Pose& child_pose = result->poses[node];
      for (const int p : used_parents) {
        const auto& corr = correspondences.at(edge_key(p, node));
        const bool parent_is_min = (p < node);
        const teaser::Pose& parent_pose = result->poses[p];
        double residual_sum = 0.0;
        int inlier_count = 0;
        for (const auto& c : corr) {
          const int p_idx = parent_is_min ? c.first : c.second;
          const int b_idx = parent_is_min ? c.second : c.first;
          if (p_idx < 0 || p_idx >= static_cast<int>(clouds[p].size()) || b_idx < 0 ||
              b_idx >= static_cast<int>(clouds[node].size())) {
            continue;
          }
          const auto& pp = clouds[p][p_idx];
          const auto& bp = clouds[node][b_idx];
          const Eigen::Vector3d wp =
              parent_pose.R * Eigen::Vector3d(pp.x, pp.y, pp.z) + parent_pose.t;
          const Eigen::Vector3d wb =
              child_pose.R * Eigen::Vector3d(bp.x, bp.y, bp.z) + child_pose.t;
          const double r = (wp - wb).norm();
          if (r <= inlier_thresh) {
            residual_sum += r;
            ++inlier_count;
          }
        }
        result->edge_residual[edge_key(p, node)] =
            (inlier_count > 0) ? residual_sum / inlier_count : -1.0;
      }
    }
  }

  // Any non-anchor node never reached (no usable path to a posed node) stays invalid.
  for (int i = 0; i < N; ++i) {
    if (!settled[i] && !is_anchor[i]) {
      std::cerr << "[teaser::multiview] Warning: node " << i
                << " could not be aligned (no usable path to an anchor).\n";
    }
  }
}

// ============================ MULTIVIEW_METHOD::SPECTRAL_SYNC ============================

// Pull edge (p,q)'s correspondences out of the clouds as two 3-by-m point matrices. The
// correspondence map is keyed by (min,max), so which member of each pair indexes which cloud
// depends on the node ordering.
bool buildEdgePoints(const std::vector<teaser::PointCloud>& clouds,
                     const std::vector<std::pair<int, int>>& corr, int p, int q, Pts* p_pts,
                     Pts* q_pts) {
  const bool p_is_min = (p < q);
  p_pts->resize(3, static_cast<Eigen::Index>(corr.size()));
  q_pts->resize(3, static_cast<Eigen::Index>(corr.size()));
  Eigen::Index col = 0;
  for (const auto& c : corr) {
    const int pi = p_is_min ? c.first : c.second;
    const int qi = p_is_min ? c.second : c.first;
    if (pi < 0 || pi >= static_cast<int>(clouds[p].size()) || qi < 0 ||
        qi >= static_cast<int>(clouds[q].size())) {
      continue; // skip malformed correspondence
    }
    const auto& pp = clouds[p][pi];
    const auto& qq = clouds[q][qi];
    p_pts->col(col) << pp.x, pp.y, pp.z;
    q_pts->col(col) << qq.x, qq.y, qq.z;
    ++col;
  }
  p_pts->conservativeResize(3, col);
  q_pts->conservativeResize(3, col);
  return col > 0;
}

// Per-edge outlier pruning, mirroring registration.cc: keep the TIMs whose lengths agree to within
// the noise bound, then take the maximum clique of the graph they induce over correspondences.
// Returns the surviving correspondence indices, or empty when the edge is unusable.
std::vector<int> selectCliqueInliers(const Pts& p_tims, const Pts& q_tims,
                                     const Eigen::Matrix<int, 2, Eigen::Dynamic>& tim_map,
                                     int num_corr, double beta,
                                     const teaser::RobustRegistrationSolver::Params& params) {
  std::vector<int> clique;
  if (params.inlier_selection_mode ==
      teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE) {
    clique.resize(num_corr);
    std::iota(clique.begin(), clique.end(), 0);
    return clique;
  }

  teaser::Graph inlier_graph;
  inlier_graph.populateVertices(num_corr);
  for (Eigen::Index k = 0; k < p_tims.cols(); ++k) {
    // Scale consistency: a rigid transform preserves TIM length, so a correspondence pair whose
    // two TIM lengths disagree by more than the noise bound cannot both be inliers.
    if (std::abs(q_tims.col(k).norm() - p_tims.col(k).norm()) <= beta) {
      inlier_graph.addEdge(tim_map(0, k), tim_map(1, k));
    }
  }

  teaser::MaxCliqueSolver::Params clique_params;
  switch (params.inlier_selection_mode) {
  case teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT:
    clique_params.solver_mode = teaser::MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_EXACT;
    break;
  case teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU:
    clique_params.solver_mode = teaser::MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_HEU;
    break;
  default:
    clique_params.solver_mode = teaser::MaxCliqueSolver::CLIQUE_SOLVER_MODE::KCORE_HEU;
    break;
  }
  clique_params.time_limit = params.max_clique_time_limit;
  clique_params.kcore_heuristic_threshold = params.kcore_heuristic_threshold;
  clique_params.num_threads = params.max_clique_num_threads;

  teaser::MaxCliqueSolver clique_solver(clique_params);
  clique = clique_solver.findMaxClique(inlier_graph);
  std::sort(clique.begin(), clique.end());
  return clique;
}

// Reduce one graph edge to a PreparedEdge. `tim_helper` only supplies computeTIMs, which is a
// non-static member. Returns false if the edge cannot support a rotation estimate.
bool prepareEdge(teaser::RobustRegistrationSolver& tim_helper,
                 const teaser::RobustRegistrationSolver::Params& params, double beta, int p, int q,
                 double confidence, const Pts& p_all, const Pts& q_all, PreparedEdge* out) {
  if (p_all.cols() < 3) {
    return false; // too few correspondences for a TIM-based rotation
  }
  Eigen::Matrix<int, 2, Eigen::Dynamic> tim_map, unused_map;
  const Pts p_tims_all = tim_helper.computeTIMs(p_all, &tim_map);
  const Pts q_tims_all = tim_helper.computeTIMs(q_all, &unused_map);

  const std::vector<int> clique = selectCliqueInliers(
      p_tims_all, q_tims_all, tim_map, static_cast<int>(p_all.cols()), beta, params);
  if (clique.size() < 3) {
    return false;
  }

  out->p = p;
  out->q = q;
  out->confidence = confidence;
  out->p_pts.resize(3, static_cast<Eigen::Index>(clique.size()));
  out->q_pts.resize(3, static_cast<Eigen::Index>(clique.size()));
  for (size_t k = 0; k < clique.size(); ++k) {
    out->p_pts.col(static_cast<Eigen::Index>(k)) = p_all.col(clique[k]);
    out->q_pts.col(static_cast<Eigen::Index>(k)) = q_all.col(clique[k]);
  }
  out->p_tims = tim_helper.computeTIMs(out->p_pts, &unused_map);
  out->q_tims = tim_helper.computeTIMs(out->q_pts, &unused_map);
  return out->p_tims.cols() > 0;
}

// GNC-TLS weight update on PRE-NORMALIZED squared residuals (r2 = r^2 / sigma^2). Normalizing up
// front makes the noise bound 1, so the thresholds are pure functions of mu and one update serves
// both stages even though their noise bounds differ. Mirrors registration.cc:928-941, including
// accumulating the cost with the PREVIOUS weights.
double gncUpdateWeights(const std::vector<RowVec>& r2, double mu, std::vector<RowVec>* w) {
  const double th1 = (mu + 1) / mu; // above this: outlier, weight 0
  const double th2 = mu / (mu + 1); // below this: inlier, weight 1
  double cost = 0;
  for (size_t e = 0; e < r2.size(); ++e) {
    for (Eigen::Index j = 0; j < r2[e].cols(); ++j) {
      cost += (*w)[e](j) * r2[e](j);
      if (r2[e](j) >= th1) {
        (*w)[e](j) = 0;
      } else if (r2[e](j) <= th2) {
        (*w)[e](j) = 1;
      } else {
        (*w)[e](j) = std::sqrt(mu * (mu + 1) / r2[e](j)) - mu;
      }
    }
  }
  return cost;
}

// Shared tail of both GNC loops: initialize mu from the largest residual on the first pass, update
// the weights, anneal, and report whether the cost has converged. Returns false to stop.
bool gncStep(const std::vector<RowVec>& r2, int iter,
             const teaser::RobustRegistrationSolver::Params& params, double* mu, double* prev_cost,
             std::vector<RowVec>* w) {
  if (iter == 0) {
    double max_r2 = 0;
    for (const auto& e : r2) {
      if (e.cols() > 0) {
        max_r2 = std::max(max_r2, e.maxCoeff());
      }
    }
    *mu = 1.0 / (2.0 * max_r2 - 1.0);
    if (*mu <= 0) {
      return false; // residuals already tiny: the surrogate is the true cost, nothing to anneal
    }
  }
  const double cost = gncUpdateWeights(r2, *mu, w);
  const double cost_diff = std::abs(cost - *prev_cost);
  *prev_cost = cost;
  *mu *= params.rotation_gnc_factor;
  return cost_diff >= params.rotation_cost_threshold;
}

// Shared core of the multi-scan drivers. Split the undirected graph edges into connected components
// (warning if more than one). A reference order fixes each component's anchor (its earliest node =
// identity pose) and the sweep order: if `order` is non-empty it is the caller-provided topological
// order (rank = position in it; nodes absent from it are ranked after all listed ones); otherwise
// the anchor is the highest-`anchor_score` node and the reference is a BFS discovery order from it.
// Poses then propagate in a wavefront over the reference order, aligning each node to ALL of its
// already-posed neighbors (child = target, parents = fixed sources); a node whose neighbors are all
// posed later is picked up in a subsequent sweep (BFS fallback).
teaser::MultiScanResult alignAlongGraph(
    const std::vector<teaser::PointCloud>& clouds,
    const std::vector<std::pair<int, int>>& graph_edges, const std::vector<double>& anchor_score,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const teaser::RobustRegistrationSolver::Params& params, const std::vector<int>& order,
    const std::map<std::pair<int, int>, double>& edge_weight,
    const teaser::UprightPrior& upright) {
  const int N = static_cast<int>(clouds.size());
  teaser::MultiScanResult result;
  result.poses.assign(N, teaser::Pose{});
  result.valid.assign(N, false);
  result.component.assign(N, -1);
  if (N == 0) {
    return result;
  }

  auto edge_key = [](int a, int b) { return std::make_pair(std::min(a, b), std::max(a, b)); };
  auto valid_edge = [N](const std::pair<int, int>& e) {
    return e.first >= 0 && e.second >= 0 && e.first < N && e.second < N && e.first != e.second;
  };

  // Deduplicated undirected adjacency + union-find components over the graph edges.
  std::vector<int> uf(N);
  std::iota(uf.begin(), uf.end(), 0);
  auto find = [&uf](int x) {
    while (uf[x] != x) {
      uf[x] = uf[uf[x]];
      x = uf[x];
    }
    return x;
  };
  std::vector<std::vector<int>> adj(N);
  std::set<std::pair<int, int>> edges;
  for (const auto& e : graph_edges) {
    if (!valid_edge(e)) {
      continue;
    }
    const auto k = edge_key(e.first, e.second);
    if (!edges.insert(k).second) {
      continue; // drop duplicate / parallel edges
    }
    adj[k.first].push_back(k.second);
    adj[k.second].push_back(k.first);
    const int ra = find(k.first);
    const int rb = find(k.second);
    if (ra != rb) {
      uf[rb] = ra;
    }
  }
  std::map<int, int> root_to_comp;
  for (int i = 0; i < N; ++i) {
    const int r = find(i);
    if (root_to_comp.find(r) == root_to_comp.end()) {
      root_to_comp[r] = static_cast<int>(root_to_comp.size());
    }
  }
  result.num_components = static_cast<int>(root_to_comp.size());
  for (int i = 0; i < N; ++i) {
    result.component[i] = root_to_comp[find(i)];
  }
  if (result.num_components > 1) {
    std::cerr << "[teaser::multiview] Warning: graph has " << result.num_components
              << " connected components; aligning each independently.\n";
  }

  // Reference order + anchors. With a caller order, rank = position in it (unlisted nodes ranked
  // after all listed ones); the anchor of each component is its earliest (min-ref) node. Otherwise
  // the anchor is the highest-anchor_score node and ref is a BFS discovery order from it.
  std::vector<int> ref(N, 0);
  std::vector<int> anchor(result.num_components, -1);
  if (!order.empty()) {
    std::vector<int> rank(N, -1);
    int pos = 0;
    for (const int node : order) {
      if (node >= 0 && node < N && rank[node] < 0) {
        rank[node] = pos++;
      }
    }
    for (int i = 0; i < N; ++i) {
      ref[i] = (rank[i] >= 0) ? rank[i] : (N + i); // unlisted nodes ranked after all listed ones
    }
    for (int i = 0; i < N; ++i) {
      const int c = result.component[i];
      if (anchor[c] < 0 || ref[i] < ref[anchor[c]]) {
        anchor[c] = i;
      }
    }
  } else {
    for (int i = 0; i < N; ++i) {
      const int c = result.component[i];
      if (anchor[c] < 0 || anchor_score[i] > anchor_score[anchor[c]]) {
        anchor[c] = i;
      }
    }
    int counter = 0;
    std::vector<char> visited(N, 0);
    for (int c = 0; c < result.num_components; ++c) {
      const int root = anchor[c];
      if (root < 0 || visited[root]) {
        continue;
      }
      std::queue<int> q;
      q.push(root);
      visited[root] = 1;
      ref[root] = counter++;
      while (!q.empty()) {
        const int u = q.front();
        q.pop();
        for (const int v : adj[u]) {
          if (!visited[v]) {
            visited[v] = 1;
            ref[v] = counter++;
            q.push(v);
          }
        }
      }
    }
  }

  // Anchors get the identity pose. An isolated node is its own component's anchor, so it stays
  // valid with an identity pose, as before.
  std::vector<char> is_anchor(N, 0);
  for (int c = 0; c < result.num_components; ++c) {
    if (anchor[c] >= 0) {
      result.poses[anchor[c]] = teaser::Pose{};
      result.valid[anchor[c]] = true;
      is_anchor[anchor[c]] = 1;
    }
  }

  // Everything above -- components, anchors, reference order -- is shared. The two methods diverge
  // only in how they turn the graph into poses.
  if (params.multiview_method ==
      teaser::RobustRegistrationSolver::MULTIVIEW_METHOD::DAG_PROPAGATION) {
    alignByPropagation(clouds, adj, ref, correspondences, params, edge_weight, is_anchor, &result);
    return result;
  }

  // ---------- Per-edge preprocessing ----------
  // Both stages consume the same max-clique-pruned correspondences, so prune once up front.
  //
  // The noise bounds are the SYMMETRIC compositions. The pairwise pipeline treats the source cloud
  // as exact and puts all noise on the destination, hence its factor of 2; here both scans' poses
  // are unknown and both sides are noisy measurements, so both contribute. Rotation consumes TIMs
  // (a difference of two points each way, 2*delta per scan); translation consumes raw points
  // (delta per scan).
  const double delta = params.noise_bound;
  const double sqrt_cbar2 = std::sqrt(std::max(0.0, params.cbar2));
  const double clique_beta = teaser::timResidualNoiseBound(delta, delta) * sqrt_cbar2;
  const double sigma_rot_sq = std::pow(teaser::timResidualNoiseBound(delta, delta), 2);
  const double sigma_trans_sq = std::pow(teaser::pointResidualNoiseBound(delta, delta), 2);

  teaser::RobustRegistrationSolver tim_helper(params); // used only for computeTIMs
  std::vector<PreparedEdge> prepared;
  prepared.reserve(edges.size());
  for (const auto& key : edges) {
    const auto it = correspondences.find(key);
    if (it == correspondences.end() || it->second.empty()) {
      continue;
    }
    const auto wit = edge_weight.find(key);
    const double confidence = (wit != edge_weight.end()) ? wit->second : 1.0;
    if (!(confidence > 0)) {
      continue; // a zero-weight edge contributes nothing to either cost
    }
    Pts p_all, q_all;
    if (!buildEdgePoints(clouds, it->second, key.first, key.second, &p_all, &q_all)) {
      continue;
    }
    PreparedEdge pe;
    if (prepareEdge(tim_helper, params, clique_beta, key.first, key.second, confidence, p_all,
                    q_all, &pe)) {
      prepared.push_back(std::move(pe));
    }
  }

  // Upright prior. Gravity is per-scan data; when absent everything below is a no-op and the solver
  // behaves exactly as it did before the prior existed.
  teaser::RotationSyncParams rot_sync_params;
  rot_sync_params.world_up = upright.world_up;
  rot_sync_params.upright_prior_eta = upright.virtual_node_eta;
  const std::vector<Eigen::Vector3d>* gravity_ptr =
      upright.gravity.empty() ? nullptr : &upright.gravity;
  if (gravity_ptr != nullptr && static_cast<int>(upright.gravity.size()) != N) {
    std::cerr << "[teaser::multiview] Warning: UprightPrior::gravity has "
              << upright.gravity.size() << " entries but there are " << N
              << " scans; ignoring the upright prior.\n";
    gravity_ptr = nullptr;
  }
  std::vector<char> node_upright(N, 0);
  std::vector<double> tilt_error;

  std::vector<Eigen::Matrix3d> R(N, Eigen::Matrix3d::Identity());
  std::vector<Eigen::Vector3d> t(N, Eigen::Vector3d::Zero());
  std::vector<bool> rot_valid(N, false);
  std::vector<bool> trans_valid(N, false);

  // ---------- Rotation: graph-level GNC-TLS ----------
  // Every node's rotation is solved at once each iteration, so a loop closure redistributes its
  // error around the whole cycle instead of dumping it on the last edge, and an edge that
  // disagrees with the rest of the graph gets down-weighted by their consensus.
  std::vector<RowVec> w_rot(prepared.size());
  for (size_t e = 0; e < prepared.size(); ++e) {
    w_rot[e] = RowVec::Ones(1, prepared[e].p_tims.cols());
  }
  {
    std::vector<RowVec> r2(prepared.size());
    double mu = 1.0;
    double prev_cost = std::numeric_limits<double>::infinity();
    for (size_t iter = 0; iter < params.rotation_max_iterations; ++iter) {
      std::vector<teaser::RotationSyncEdge> sync_edges(prepared.size());
      for (size_t e = 0; e < prepared.size(); ++e) {
        sync_edges[e].p = prepared[e].p;
        sync_edges[e].q = prepared[e].q;
        sync_edges[e].p_tims = prepared[e].p_tims;
        sync_edges[e].q_tims = prepared[e].q_tims;
        sync_edges[e].w = w_rot[e];
        // Fold the surviving weight mass into the edge confidence. The line-process weights only
        // reshape M *within* an edge, and Rhat is M's polar factor, which is invariant to scaling
        // M -- so without this a uniformly down-weighted edge keeps its full influence on B and the
        // graph has no way to overrule an edge that is coherently wrong. This is the only channel
        // that can. (synchronizeTranslations already folds the mass in itself.)
        sync_edges[e].confidence = prepared[e].confidence * w_rot[e].sum();
      }
      // Passing the current estimate keeps a node that has just lost all its edges to the weight
      // update from snapping back to the identity.
      const teaser::RotationSyncResult sync = teaser::synchronizeRotations(
          N, sync_edges, rot_sync_params, &R, /*node_noise_bounds=*/nullptr, gravity_ptr);
      R = sync.rotations;
      rot_valid = sync.valid;
      // Remember per node whether its component's gauge was made upright. The sync helper labels
      // components itself (edgeless nodes get -1), which need not match this driver's labeling, so
      // carry the flag per node rather than per component id.
      if (gravity_ptr != nullptr) {
        for (int i = 0; i < N; ++i) {
          const int sc = sync.component[i];
          node_upright[i] = (sc >= 0 && sc < static_cast<int>(sync.gauge_upright.size()) &&
                             sync.gauge_upright[sc])
                                ? 1
                                : 0;
        }
        tilt_error = sync.tilt_error;
      }

      for (size_t e = 0; e < prepared.size(); ++e) {
        const Pts diff =
            R[prepared[e].p] * prepared[e].p_tims - R[prepared[e].q] * prepared[e].q_tims;
        r2[e] = diff.colwise().squaredNorm() / sigma_rot_sq;
      }
      if (!gncStep(r2, static_cast<int>(iter), params, &mu, &prev_cost, &w_rot)) {
        break;
      }
    }
  }

  // ---------- Translation: graph-level GNC-TLS, given the rotations ----------
  std::vector<RowVec> w_trans(prepared.size());
  for (size_t e = 0; e < prepared.size(); ++e) {
    w_trans[e] = RowVec::Ones(1, prepared[e].p_pts.cols());
  }
  {
    std::vector<RowVec> r2(prepared.size());
    double mu = 1.0;
    double prev_cost = std::numeric_limits<double>::infinity();
    for (size_t iter = 0; iter < params.rotation_max_iterations; ++iter) {
      std::vector<teaser::TranslationSyncEdge> sync_edges(prepared.size());
      for (size_t e = 0; e < prepared.size(); ++e) {
        sync_edges[e].p = prepared[e].p;
        sync_edges[e].q = prepared[e].q;
        sync_edges[e].p_pts = prepared[e].p_pts;
        sync_edges[e].q_pts = prepared[e].q_pts;
        sync_edges[e].w = w_trans[e];
        sync_edges[e].confidence = prepared[e].confidence;
      }
      const teaser::TranslationSyncResult sync = teaser::synchronizeTranslations(
          N, sync_edges, R, teaser::TranslationSyncParams(), &t);
      t = sync.translations;
      trans_valid = sync.valid;

      for (size_t e = 0; e < prepared.size(); ++e) {
        const Pts lhs = (R[prepared[e].p] * prepared[e].p_pts).colwise() + t[prepared[e].p];
        const Pts rhs = (R[prepared[e].q] * prepared[e].q_pts).colwise() + t[prepared[e].q];
        r2[e] = (lhs - rhs).colwise().squaredNorm() / sigma_trans_sq;
      }
      if (!gncStep(r2, static_cast<int>(iter), params, &mu, &prev_cost, &w_trans)) {
        break;
      }
    }
  }

  // ---------- Joint Levenberg-Marquardt refinement (optional) ----------
  // The spectral rotation and Laplacian translation solves are relaxations that each discard
  // information (per-edge anisotropy; the rotation->translation coupling). A local LM refinement of
  // the true raw-point objective, seeded by (R, t) above, recovers it -- see pose_refine.h. The LM
  // solve only commits cost-decreasing steps, so it never returns poses worse than the seed. It
  // runs in the sync gauge (each component's lowest-indexed node held fixed), composing with the
  // re-gauge below unchanged. Gated by params.multiview_refine_iterations (0 = off).
  if (params.multiview_refine_iterations > 0) {
    std::vector<teaser::PoseRefineEdge> refine_edges(prepared.size());
    for (size_t e = 0; e < prepared.size(); ++e) {
      refine_edges[e].p = prepared[e].p;
      refine_edges[e].q = prepared[e].q;
      refine_edges[e].p_pts = prepared[e].p_pts;
      refine_edges[e].q_pts = prepared[e].q_pts;
      refine_edges[e].w = w_trans[e]; // converged point-level GNC weights
      refine_edges[e].edge_weight = prepared[e].confidence / sigma_trans_sq;
    }
    teaser::PoseRefineParams rp;
    rp.max_iterations = params.multiview_refine_iterations;
    const teaser::PoseRefineResult refined = teaser::refinePoses(N, refine_edges, R, t, rp);
    R = refined.rotations;
    t = refined.translations;
    std::cerr << "[teaser::multiview] refinement moved poses by "
              << refined.avg_translation_change << " (translation, avg) / "
              << refined.avg_rotation_change << " rad (rotation, avg).\n";
  }

  // ---------- Re-gauge to each component's anchor ----------
  // Both helpers pin their component's lowest-indexed node; the multiview anchor rule may pick a
  // different one, so re-express every pose in the anchor's frame. The anchor lands on exactly
  // identity / zero.
  for (int c = 0; c < result.num_components; ++c) {
    const int a = anchor[c];
    if (a < 0) {
      continue;
    }
    // When the rotation stage already fixed this component's gauge by gravity, re-expressing
    // everything in the anchor's frame would UNDO that and hand back a tilted map. Keep the upright
    // frame and only zero the anchor's translation (the translation gauge is additive, so it is
    // independent of the rotation gauge).
    const Eigen::Matrix3d Ra =
        node_upright[a] ? Eigen::Matrix3d::Identity() : Eigen::Matrix3d(R[a].transpose());
    const Eigen::Vector3d ta = t[a];
    for (int i = 0; i < N; ++i) {
      if (result.component[i] != c) {
        continue;
      }
      result.poses[i].R = Ra * R[i];
      result.poses[i].t = Ra * (t[i] - ta);
      result.valid[i] = is_anchor[i] || (rot_valid[i] && trans_valid[i]);
    }
  }
  if (!tilt_error.empty()) {
    int worst = -1;
    double worst_tilt = -1.0;
    for (int i = 0; i < N; ++i) {
      if (result.valid[i] && i < static_cast<int>(tilt_error.size()) &&
          std::isfinite(tilt_error[i]) && tilt_error[i] > worst_tilt) {
        worst_tilt = tilt_error[i];
        worst = i;
      }
    }
    if (worst >= 0) {
      std::cerr << "[teaser::multiview] upright prior: worst residual tilt "
                << worst_tilt * 180.0 / M_PI << " deg at scan " << worst
                << (worst_tilt > M_PI / 2
                        ? " -- that scan is UPSIDE DOWN; gravity cannot repair it (yaw is"
                          " unconstrained), re-estimate it from its neighbours.\n"
                        : ".\n");
    }
  }
  for (int i = 0; i < N; ++i) {
    if (!result.valid[i]) {
      std::cerr << "[teaser::multiview] Warning: node " << i
                << " could not be aligned (no usable edge survived).\n";
    }
  }

  // ---------- Per-edge fit quality ----------
  // Mean world-frame residual over each edge's inlier correspondences (those consistent with the
  // recovered poses within the noise bound). Every surviving edge gets an entry now -- with joint
  // synchronization there is no distinction between tree edges and loop closures.
  const double inlier_thresh = 2.0 * params.noise_bound * sqrt_cbar2;
  for (const auto& pe : prepared) {
    if (!result.valid[pe.p] || !result.valid[pe.q]) {
      continue;
    }
    const auto key = edge_key(pe.p, pe.q);
    const auto& corr = correspondences.at(key);
    const bool p_is_min = (pe.p < pe.q);
    double residual_sum = 0.0;
    int inlier_count = 0;
    for (const auto& c : corr) {
      const int pi = p_is_min ? c.first : c.second;
      const int qi = p_is_min ? c.second : c.first;
      if (pi < 0 || pi >= static_cast<int>(clouds[pe.p].size()) || qi < 0 ||
          qi >= static_cast<int>(clouds[pe.q].size())) {
        continue;
      }
      const auto& pp = clouds[pe.p][pi];
      const auto& qq = clouds[pe.q][qi];
      const Eigen::Vector3d wp =
          result.poses[pe.p].R * Eigen::Vector3d(pp.x, pp.y, pp.z) + result.poses[pe.p].t;
      const Eigen::Vector3d wq =
          result.poses[pe.q].R * Eigen::Vector3d(qq.x, qq.y, qq.z) + result.poses[pe.q].t;
      const double r = (wp - wq).norm();
      if (r <= inlier_thresh) {
        residual_sum += r;
        ++inlier_count;
      }
    }
    result.edge_residual[key] = (inlier_count > 0) ? residual_sum / inlier_count : -1.0;
  }

  return result;
}

} // namespace

teaser::MultiScanResult teaser::alignMultiScan(
    const std::vector<teaser::PointCloud>& clouds, const teaser::Graph& adjacency,
    const std::map<std::pair<int, int>, double>& edge_weights,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const teaser::RobustRegistrationSolver::Params& params,
    const teaser::UprightPrior& upright) {
  const int N = static_cast<int>(clouds.size());
  teaser::MultiScanResult result;
  result.poses.assign(N, teaser::Pose{});
  result.valid.assign(N, false);
  result.component.assign(N, -1);
  if (N == 0) {
    return result;
  }

  auto edge_key = [](int a, int b) { return std::make_pair(std::min(a, b), std::max(a, b)); };

  // --- 1. Weighted undirected edges (weight supplied by the caller). ---
  std::vector<teaser::WeightedEdge> weighted_edges;
  const auto adj = adjacency.getAdjList();
  const int V = std::min<int>(N, static_cast<int>(adj.size()));
  for (int i = 0; i < V; ++i) {
    for (const int j : adj[i]) {
      if (j <= i || j >= N) {
        continue; // visit each undirected edge once, ignore out-of-range vertices
      }
      const auto it = edge_weights.find(edge_key(i, j));
      const double w = (it != edge_weights.end()) ? it->second : 0.0;
      weighted_edges.push_back({i, j, w});
    }
  }
  // --- 2. Choose the edge set to align over. ---
  // Anchors are always scored by total incident edge weight over the whole adjacency graph.
  // `edge_weights` is documented as a spanning-tree weight independent of the correspondences, so it
  // is deliberately NOT reused as a per-edge cost weight in either method.
  std::vector<double> anchor_score(N, 0.0);
  for (const auto& e : weighted_edges) {
    anchor_score[e.u] += e.weight;
    anchor_score[e.v] += e.weight;
  }

  std::vector<std::pair<int, int>> selected_edges;
  if (params.multiview_method ==
      teaser::RobustRegistrationSolver::MULTIVIEW_METHOD::DAG_PROPAGATION) {
    // Propagation walks a tree: prune to the maximum spanning tree, as documented.
    const std::vector<teaser::WeightedEdge> forest =
        teaser::kruskalSpanningTree(N, weighted_edges, /*maximum=*/true);
    selected_edges.reserve(forest.size());
    for (const auto& e : forest) {
      selected_edges.push_back({e.u, e.v});
    }
  } else {
    // Synchronization wants every edge. A tree is exactly determined, so pruning to one would just
    // reproduce what propagation already does; keeping the extra edges is what lets loop closures
    // be averaged in.
    selected_edges.reserve(weighted_edges.size());
    for (const auto& e : weighted_edges) {
      selected_edges.push_back({e.u, e.v});
    }
  }

  return alignAlongGraph(clouds, selected_edges, anchor_score, correspondences, params, /*order=*/{},
                         /*edge_weight=*/{}, upright);
}

teaser::MultiScanResult teaser::alignMultiScanWithGraph(
    const std::vector<teaser::PointCloud>& clouds,
    const std::vector<std::pair<int, int>>& graph_edges,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const teaser::RobustRegistrationSolver::Params& params, const std::vector<int>& order,
    const std::vector<double>& edge_weights, const teaser::UprightPrior& upright) {
  const int N = static_cast<int>(clouds.size());
  // Score anchors by degree: the most-connected node in the caller-provided graph is the root.
  // (Used only when no explicit order is given; an order overrides anchor selection.)
  std::vector<double> anchor_score(N, 0.0);
  for (const auto& e : graph_edges) {
    if (e.first >= 0 && e.first < N) {
      anchor_score[e.first] += 1.0;
    }
    if (e.second >= 0 && e.second < N) {
      anchor_score[e.second] += 1.0;
    }
  }

  // Build a (min,max)->weight lookup from the per-edge weights (aligned with graph_edges). First
  // occurrence wins on duplicate edges; edges without a weight default to 1.0.
  std::map<std::pair<int, int>, double> edge_weight;
  if (!edge_weights.empty()) {
    assert(edge_weights.size() == graph_edges.size());
    for (size_t i = 0; i < graph_edges.size() && i < edge_weights.size(); ++i) {
      const auto k = std::make_pair(std::min(graph_edges[i].first, graph_edges[i].second),
                                    std::max(graph_edges[i].first, graph_edges[i].second));
      edge_weight.emplace(k, edge_weights[i]); // emplace keeps the first occurrence
    }
  }
  return alignAlongGraph(clouds, graph_edges, anchor_score, correspondences, params, order,
                         edge_weight, upright);
}
