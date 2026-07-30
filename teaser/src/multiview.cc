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
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
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
  // the world frame: src_world = R_i * a^i + t_i, dst_local = b^i.
  Eigen::Matrix<double, 3, Eigen::Dynamic> src_world(3, total);
  Eigen::Matrix<double, 3, Eigen::Dynamic> dst_local(3, total);
  Eigen::Index offset = 0;
  for (const auto& edge : edges) {
    const Eigen::Index k = edge.src.cols();
    if (k == 0 || edge.src.cols() != edge.dst.cols()) {
      continue;
    }
    src_world.middleCols(offset, k) = (edge.R_i * edge.src).colwise() + edge.t_i;
    dst_local.middleCols(offset, k) = edge.dst;
    offset += k;
  }

  // Solve the robust registration with scale fixed to 1. With src = world and dst = B-local, the
  // solver returns the world->B transform, i.e. rotation = R_B^T, translation = -R_B^T * t_B.
  teaser::RobustRegistrationSolver::Params p = params_;
  p.estimate_scaling = false;
  teaser::RobustRegistrationSolver solver(p);
  teaser::RegistrationSolution sol = solver.solve(src_world, dst_local);

  // Invert the world->B transform to recover B's global (B->world) pose.
  teaser::RegistrationSolution out;
  out.valid = sol.valid;
  out.scale = 1.0;
  out.rotation = sol.rotation.transpose();           // R_B = (R_B^T)^T
  out.translation = -out.rotation * sol.translation; // t_B = -R_B * (-R_B^T t_B)
  return out;
}

namespace {

// Shared core of the multi-scan drivers. Given a set of undirected tree/forest edges and a
// per-node anchor score, split into connected components (warning if more than one), pick each
// component's anchor (max score, ties -> smallest index) as the identity pose, and propagate poses
// outward along the tree in topological order (child = target, parent = fixed source).
teaser::MultiScanResult alignAlongForest(
    const std::vector<teaser::PointCloud>& clouds,
    const std::vector<std::pair<int, int>>& forest_edges, const std::vector<double>& anchor_score,
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
  auto valid_edge = [N](const std::pair<int, int>& e) {
    return e.first >= 0 && e.second >= 0 && e.first < N && e.second < N && e.first != e.second;
  };

  // Connected components via union-find over the forest edges.
  std::vector<int> uf(N);
  std::iota(uf.begin(), uf.end(), 0);
  auto find = [&uf](int x) {
    while (uf[x] != x) {
      uf[x] = uf[uf[x]];
      x = uf[x];
    }
    return x;
  };
  for (const auto& e : forest_edges) {
    if (!valid_edge(e)) {
      continue;
    }
    const int ra = find(e.first);
    const int rb = find(e.second);
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

  // Anchor per component: max anchor_score, ties -> smallest index.
  std::vector<int> anchor(result.num_components, -1);
  for (int i = 0; i < N; ++i) {
    const int c = result.component[i];
    if (anchor[c] < 0 || anchor_score[i] > anchor_score[anchor[c]]) {
      anchor[c] = i;
    }
  }

  // Root each tree at its anchor (BFS), orient parent->child, then topological order.
  std::vector<std::vector<int>> tree_adj(N);
  for (const auto& e : forest_edges) {
    if (!valid_edge(e)) {
      continue;
    }
    tree_adj[e.first].push_back(e.second);
    tree_adj[e.second].push_back(e.first);
  }
  std::vector<int> tree_parent(N, -1);
  std::vector<std::pair<int, int>> directed;
  std::vector<char> visited(N, 0);
  for (int c = 0; c < result.num_components; ++c) {
    const int root = anchor[c];
    if (root < 0 || visited[root]) {
      continue;
    }
    std::queue<int> q;
    q.push(root);
    visited[root] = 1;
    while (!q.empty()) {
      const int u = q.front();
      q.pop();
      for (const int v : tree_adj[u]) {
        if (!visited[v]) {
          visited[v] = 1;
          tree_parent[v] = u;
          directed.push_back({u, v});
          q.push(v);
        }
      }
    }
  }
  const std::vector<int> order = teaser::topologicalSort(N, directed);

  // Anchors get the identity pose.
  for (int c = 0; c < result.num_components; ++c) {
    if (anchor[c] >= 0) {
      result.poses[anchor[c]] = teaser::Pose{};
      result.valid[anchor[c]] = true;
    }
  }

  // Propagate poses along the tree in topological order.
  teaser::MultiviewSolver solver(params);
  for (const int node : order) {
    const int p = tree_parent[node];
    if (p < 0) {
      continue; // anchor/root or isolated singleton (already identity)
    }
    if (!result.valid[p]) {
      std::cerr << "[teaser::multiview] Warning: node " << node << " skipped (parent " << p
                << " has no valid pose).\n";
      result.valid[node] = false;
      continue;
    }
    const auto it = correspondences.find(edge_key(p, node));
    if (it == correspondences.end() || it->second.empty()) {
      std::cerr << "[teaser::multiview] Warning: node " << node
                << " skipped (no correspondences on tree edge to parent " << p << ").\n";
      result.valid[node] = false;
      continue;
    }
    const auto& corr = it->second;
    const bool parent_is_min = (p < node);

    // Build one NeighborEdge: parent = source (fixed pose), child (node) = target/dst.
    teaser::NeighborEdge edge;
    edge.R_i = result.poses[p].R;
    edge.t_i = result.poses[p].t;
    edge.src.resize(3, static_cast<Eigen::Index>(corr.size()));
    edge.dst.resize(3, static_cast<Eigen::Index>(corr.size()));
    Eigen::Index col = 0;
    for (const auto& c : corr) {
      const int p_idx = parent_is_min ? c.first : c.second;  // index into parent cloud
      const int b_idx = parent_is_min ? c.second : c.first;  // index into child cloud
      if (p_idx < 0 || p_idx >= static_cast<int>(clouds[p].size()) || b_idx < 0 ||
          b_idx >= static_cast<int>(clouds[node].size())) {
        continue; // skip malformed correspondence
      }
      const auto& pp = clouds[p][p_idx];
      const auto& bp = clouds[node][b_idx];
      edge.src.col(col) << pp.x, pp.y, pp.z;
      edge.dst.col(col) << bp.x, bp.y, bp.z;
      ++col;
    }
    edge.src.conservativeResize(3, col);
    edge.dst.conservativeResize(3, col);

    const teaser::RegistrationSolution sol = solver.solveNodePose({edge});
    result.poses[node].R = sol.rotation;
    result.poses[node].t = sol.translation;
    result.valid[node] = sol.valid;
    if (!sol.valid) {
      std::cerr << "[teaser::multiview] Warning: alignment failed for node " << node << ".\n";
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

  return alignAlongForest(clouds, forest_edges, anchor_score, correspondences, params);
}

teaser::MultiScanResult teaser::alignMultiScanWithTree(
    const std::vector<teaser::PointCloud>& clouds,
    const std::vector<std::pair<int, int>>& tree_edges,
    const std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>& correspondences,
    const teaser::RobustRegistrationSolver::Params& params) {
  const int N = static_cast<int>(clouds.size());
  // Score anchors by tree degree: the most-connected node in the caller-provided tree.
  std::vector<double> anchor_score(N, 0.0);
  for (const auto& e : tree_edges) {
    if (e.first >= 0 && e.first < N) {
      anchor_score[e.first] += 1.0;
    }
    if (e.second >= 0 && e.second < N) {
      anchor_score[e.second] += 1.0;
    }
  }
  return alignAlongForest(clouds, tree_edges, anchor_score, correspondences, params);
}
