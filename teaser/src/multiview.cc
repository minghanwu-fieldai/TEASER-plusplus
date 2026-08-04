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
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <utility>

#include "teaser/registration.h"

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
    const int p_idx = parent_is_min ? c.first : c.second;  // index into parent cloud
    const int b_idx = parent_is_min ? c.second : c.first;  // index into child cloud
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
    const std::map<std::pair<int, int>, double>& edge_weight) {
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

  // Sweep order: nodes sorted by ref (the caller order, or the BFS discovery order).
  std::vector<int> proc(N);
  std::iota(proc.begin(), proc.end(), 0);
  std::stable_sort(proc.begin(), proc.end(), [&ref](int a, int b) { return ref[a] < ref[b]; });

  // Anchors get the identity pose (and seed the wavefront).
  std::vector<char> is_anchor(N, 0);
  std::vector<char> settled(N, 0);
  for (int c = 0; c < result.num_components; ++c) {
    if (anchor[c] >= 0) {
      result.poses[anchor[c]] = teaser::Pose{};
      result.valid[anchor[c]] = true;
      is_anchor[anchor[c]] = 1;
      settled[anchor[c]] = 1;
    }
  }

  // Wavefront: sweep nodes in ref order to a fixpoint, aligning each not-yet-settled node that has
  // >= 1 already-posed neighbor to ALL such neighbors. Nodes with no posed neighbor yet are deferred
  // to a later sweep (BFS fallback from the anchor).
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
        if (!result.valid[p]) {
          continue; // neighbor not posed yet (or failed)
        }
        const auto it = correspondences.find(edge_key(p, node));
        if (it == correspondences.end() || it->second.empty()) {
          continue;
        }
        teaser::NeighborEdge edge;
        if (buildNeighborEdge(clouds, it->second, p, node, result.poses[p], &edge)) {
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
      result.poses[node].R = sol.rotation;
      result.poses[node].t = sol.translation;
      result.valid[node] = sol.valid;
      if (!sol.valid) {
        std::cerr << "[teaser::multiview] Warning: alignment failed for node " << node << ".\n";
        continue;
      }

      // Per-edge fit quality: mean world-frame residual over each edge's inlier correspondences
      // (those consistent with the recovered pose within the noise bound).
      const double inlier_thresh = 2.0 * params.noise_bound * std::sqrt(std::max(0.0, params.cbar2));
      const teaser::Pose& child_pose = result.poses[node];
      for (const int p : used_parents) {
        const auto& corr = correspondences.at(edge_key(p, node));
        const bool parent_is_min = (p < node);
        const teaser::Pose& parent_pose = result.poses[p];
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
          const Eigen::Vector3d wb = child_pose.R * Eigen::Vector3d(bp.x, bp.y, bp.z) + child_pose.t;
          const double r = (wp - wb).norm();
          if (r <= inlier_thresh) {
            residual_sum += r;
            ++inlier_count;
          }
        }
        result.edge_residual[edge_key(p, node)] =
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

  return result;
}

} // namespace

teaser::MultiScanResult teaser::alignMultiScan(
    const std::vector<teaser::PointCloud>& clouds, const teaser::Graph& adjacency,
    const std::map<std::pair<int, int>, double>& edge_weights,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const teaser::RobustRegistrationSolver::Params& params) {
  const int N = static_cast<int>(clouds.size());
  teaser::MultiScanResult result;
  result.poses.assign(N, teaser::Pose{});
  result.valid.assign(N, false);
  result.component.assign(N, -1);
  if (N == 0) {
    return result;
  }

  auto edge_key = [](int a, int b) { return std::make_pair(std::min(a, b), std::max(a, b)); };

  // --- 1. Weighted undirected edges (weight supplied by the caller), then MST. ---
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
  const std::vector<teaser::WeightedEdge> forest =
      teaser::kruskalSpanningTree(N, weighted_edges, /*maximum=*/true);

  // The MST edges become the propagation tree; anchors are scored by total incident edge weight.
  std::vector<std::pair<int, int>> forest_edges;
  forest_edges.reserve(forest.size());
  for (const auto& e : forest) {
    forest_edges.push_back({e.u, e.v});
  }
  std::vector<double> anchor_score(N, 0.0);
  for (const auto& e : weighted_edges) {
    anchor_score[e.u] += e.weight;
    anchor_score[e.v] += e.weight;
  }

  return alignAlongGraph(clouds, forest_edges, anchor_score, correspondences, params, /*order=*/{},
                         /*edge_weight=*/{});
}

teaser::MultiScanResult teaser::alignMultiScanWithGraph(
    const std::vector<teaser::PointCloud>& clouds,
    const std::vector<std::pair<int, int>>& graph_edges,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const teaser::RobustRegistrationSolver::Params& params, const std::vector<int>& order,
    const std::vector<double>& edge_weights) {
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
                         edge_weight);
}
