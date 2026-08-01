/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <map>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "teaser/multiview.h"
#include "teaser/registration.h"
#include "test_utils.h"

namespace {

// Deterministic RNG so the synthetic geometry is reproducible across runs.
std::mt19937& rng() {
  static std::mt19937 gen(2024);
  return gen;
}

Eigen::Matrix3d makeRotation(double ax, double ay, double az, double angle) {
  return Eigen::AngleAxisd(angle, Eigen::Vector3d(ax, ay, az).normalized()).toRotationMatrix();
}

// Sample N world points uniformly in a cube.
Eigen::Matrix<double, 3, Eigen::Dynamic> sampleWorldPoints(int N) {
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  Eigen::Matrix<double, 3, Eigen::Dynamic> W(3, N);
  for (int i = 0; i < N; ++i) {
    W.col(i) << dist(rng()), dist(rng()), dist(rng());
  }
  return W;
}

// Project world points into a scan's local frame given that scan's global pose (local -> world):
// local = R^T (world - t).
Eigen::Matrix<double, 3, Eigen::Dynamic>
worldToLocal(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
             const Eigen::Matrix<double, 3, Eigen::Dynamic>& W) {
  return R.transpose() * (W.colwise() - t);
}

teaser::RobustRegistrationSolver::Params makeParams(double noise_bound) {
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = noise_bound;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_gnc_factor = 1.4;
  params.rotation_max_iterations = 100;
  params.rotation_cost_threshold = 1e-12;
  return params;
}

// A synthetic multi-scan scene: clouds, adjacency, per-edge correspondences, and GT global poses.
struct Scene {
  std::vector<teaser::PointCloud> clouds;
  teaser::Graph adjacency;
  std::map<std::pair<int, int>, double> edge_weights;
  std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> corr;
  std::vector<teaser::Pose> gt;
};

// Build a scene from ground-truth poses (local->world) and an edge list. Each edge gets `k_in`
// shared inlier correspondences plus `k_out` gross-outlier correspondences (mismatched points).
Scene buildScene(const std::vector<teaser::Pose>& gt,
                 const std::vector<std::pair<int, int>>& edges, int k_in, int k_out) {
  const int N = static_cast<int>(gt.size());
  Scene s;
  s.gt = gt;
  std::vector<std::vector<Eigen::Vector3d>> local_pts(N);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);

  auto add_world_point = [&](int node, const Eigen::Vector3d& w) {
    local_pts[node].push_back(gt[node].R.transpose() * (w - gt[node].t));
    return static_cast<int>(local_pts[node].size()) - 1;
  };

  for (const auto& e : edges) {
    const int i = e.first, j = e.second;
    std::vector<std::pair<int, int>> pairs;
    // Inliers: the same world point observed in both scans.
    for (int k = 0; k < k_in; ++k) {
      Eigen::Vector3d w(dist(rng()), dist(rng()), dist(rng()));
      const int ii = add_world_point(i, w);
      const int jj = add_world_point(j, w);
      pairs.push_back(i < j ? std::make_pair(ii, jj) : std::make_pair(jj, ii));
    }
    // Gross outliers: unrelated points in each scan.
    for (int k = 0; k < k_out; ++k) {
      const int ii = add_world_point(i, Eigen::Vector3d(dist(rng()), dist(rng()), dist(rng())));
      const int jj = add_world_point(j, Eigen::Vector3d(dist(rng()), dist(rng()), dist(rng())));
      pairs.push_back(i < j ? std::make_pair(ii, jj) : std::make_pair(jj, ii));
    }
    s.corr[{std::min(i, j), std::max(i, j)}] = pairs;
    // Caller-supplied MST weight (uniform here; independent of the correspondences).
    s.edge_weights[{std::min(i, j), std::max(i, j)}] = 1.0;
  }

  s.adjacency.populateVertices(N);
  for (const auto& e : edges) {
    s.adjacency.addEdge(e.first, e.second);
  }
  s.clouds.resize(N);
  for (int n = 0; n < N; ++n) {
    for (const auto& p : local_pts[n]) {
      s.clouds[n].push_back({static_cast<float>(p.x()), static_cast<float>(p.y()),
                             static_cast<float>(p.z())});
    }
  }
  return s;
}

} // namespace

// Single neighbor edge, noiseless: recover the target node's global pose exactly.
TEST(MultiviewTest, SingleEdgeNoiseless) {
  Eigen::Matrix3d R_A = makeRotation(0.2, -0.5, 1.0, 0.6);
  Eigen::Vector3d t_A(1.0, -2.0, 0.5);
  Eigen::Matrix3d R_B = makeRotation(-1.0, 0.3, 0.4, -0.9);
  Eigen::Vector3d t_B(-0.5, 1.5, 2.0);

  auto W = sampleWorldPoints(40);

  teaser::NeighborEdge edge;
  edge.R_i = R_A;
  edge.t_i = t_A;
  edge.src = worldToLocal(R_A, t_A, W); // a^A in neighbor A's local frame
  edge.dst = worldToLocal(R_B, t_B, W); // b in node B's local frame

  teaser::MultiviewSolver solver(makeParams(1e-4));
  auto sol = solver.solveNodePose({edge});

  EXPECT_TRUE(sol.valid);
  EXPECT_LE(teaser::test::getAngularError(R_B, sol.rotation), 1e-5);
  EXPECT_LE((sol.translation - t_B).norm(), 1e-5);
}

// Two neighbor edges (each with its own fixed pose), noiseless: aggregation across edges.
TEST(MultiviewTest, MultiEdgeNoiseless) {
  Eigen::Matrix3d R_A = makeRotation(0.2, -0.5, 1.0, 0.6);
  Eigen::Vector3d t_A(1.0, -2.0, 0.5);
  Eigen::Matrix3d R_C = makeRotation(0.7, 0.1, -0.3, 1.2);
  Eigen::Vector3d t_C(-1.0, 0.4, -1.5);
  Eigen::Matrix3d R_B = makeRotation(-1.0, 0.3, 0.4, -0.9);
  Eigen::Vector3d t_B(-0.5, 1.5, 2.0);

  auto W_A = sampleWorldPoints(25);
  auto W_C = sampleWorldPoints(25);

  teaser::NeighborEdge edge_A;
  edge_A.R_i = R_A;
  edge_A.t_i = t_A;
  edge_A.src = worldToLocal(R_A, t_A, W_A);
  edge_A.dst = worldToLocal(R_B, t_B, W_A);

  teaser::NeighborEdge edge_C;
  edge_C.R_i = R_C;
  edge_C.t_i = t_C;
  edge_C.src = worldToLocal(R_C, t_C, W_C);
  edge_C.dst = worldToLocal(R_B, t_B, W_C);

  teaser::MultiviewSolver solver(makeParams(1e-4));
  auto sol = solver.solveNodePose({edge_A, edge_C});

  EXPECT_TRUE(sol.valid);
  EXPECT_LE(teaser::test::getAngularError(R_B, sol.rotation), 1e-5);
  EXPECT_LE((sol.translation - t_B).norm(), 1e-5);
}

// Multi-edge with a fraction of gross outliers: GNC-TLS + max-clique should still recover the pose.
TEST(MultiviewTest, MultiEdgeWithOutliers) {
  Eigen::Matrix3d R_A = makeRotation(0.2, -0.5, 1.0, 0.6);
  Eigen::Vector3d t_A(1.0, -2.0, 0.5);
  Eigen::Matrix3d R_C = makeRotation(0.7, 0.1, -0.3, 1.2);
  Eigen::Vector3d t_C(-1.0, 0.4, -1.5);
  Eigen::Matrix3d R_B = makeRotation(-1.0, 0.3, 0.4, -0.9);
  Eigen::Vector3d t_B(-0.5, 1.5, 2.0);

  const int N = 40;
  auto W_A = sampleWorldPoints(N);
  auto W_C = sampleWorldPoints(N);

  teaser::NeighborEdge edge_A;
  edge_A.R_i = R_A;
  edge_A.t_i = t_A;
  edge_A.src = worldToLocal(R_A, t_A, W_A);
  edge_A.dst = worldToLocal(R_B, t_B, W_A);

  teaser::NeighborEdge edge_C;
  edge_C.R_i = R_C;
  edge_C.t_i = t_C;
  edge_C.src = worldToLocal(R_C, t_C, W_C);
  edge_C.dst = worldToLocal(R_B, t_B, W_C);

  // Corrupt ~25% of the target measurements in each edge with random points.
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::uniform_int_distribution<int> idx(0, N - 1);
  const int num_outliers = N / 4;
  for (int o = 0; o < num_outliers; ++o) {
    edge_A.dst.col(idx(rng())) << dist(rng()), dist(rng()), dist(rng());
    edge_C.dst.col(idx(rng())) << dist(rng()), dist(rng()), dist(rng());
  }

  teaser::MultiviewSolver solver(makeParams(1e-3));
  auto sol = solver.solveNodePose({edge_A, edge_C});

  EXPECT_TRUE(sol.valid);
  EXPECT_LE(teaser::test::getAngularError(R_B, sol.rotation), 1e-2);
  EXPECT_LE((sol.translation - t_B).norm(), 1e-2);
}

// Kruskal maximum spanning tree on a small connected weighted graph.
TEST(MultiviewTest, KruskalMaximumSpanningTree) {
  // 4 vertices. Edges (weights chosen so the MaxST is unambiguous):
  //   0-1: 10, 0-2: 1, 1-2: 6, 1-3: 5, 2-3: 4
  // Maximum spanning tree picks 0-1(10), 1-2(6), 1-3(5) => total weight 21.
  std::vector<teaser::WeightedEdge> edges = {
      {0, 1, 10.0}, {0, 2, 1.0}, {1, 2, 6.0}, {1, 3, 5.0}, {2, 3, 4.0}};

  auto tree = teaser::kruskalSpanningTree(4, edges, /*maximum=*/true);

  ASSERT_EQ(tree.size(), 3u); // connected => n-1 edges
  double total = 0.0;
  for (const auto& e : tree) {
    total += e.weight;
  }
  EXPECT_DOUBLE_EQ(total, 21.0);
  // Edges are returned in descending-weight order for a maximum spanning tree.
  EXPECT_DOUBLE_EQ(tree.front().weight, 10.0);
}

// Kruskal on a disconnected graph returns a spanning forest (< n-1 edges).
TEST(MultiviewTest, KruskalSpanningForest) {
  // 5 vertices, two components {0,1,2} and {3,4}.
  std::vector<teaser::WeightedEdge> edges = {
      {0, 1, 2.0}, {1, 2, 3.0}, {3, 4, 1.0}};

  auto forest = teaser::kruskalSpanningTree(5, edges, /*maximum=*/true);

  // 3 vertices in one component (2 edges) + 2 in the other (1 edge) = 3 edges, and 3 < 5-1.
  EXPECT_EQ(forest.size(), 3u);
}

// Minimum spanning tree variant.
TEST(MultiviewTest, KruskalMinimumSpanningTree) {
  std::vector<teaser::WeightedEdge> edges = {
      {0, 1, 10.0}, {0, 2, 1.0}, {1, 2, 6.0}, {1, 3, 5.0}, {2, 3, 4.0}};

  auto tree = teaser::kruskalSpanningTree(4, edges, /*maximum=*/false);

  ASSERT_EQ(tree.size(), 3u);
  double total = 0.0;
  for (const auto& e : tree) {
    total += e.weight;
  }
  // Minimum spanning tree: 0-2(1), 2-3(4), 1-3(5) => 10.
  EXPECT_DOUBLE_EQ(total, 10.0);
}

// Topological sort of a rooted tree: every parent precedes its child; root emitted first.
TEST(MultiviewTest, TopologicalSortRootedTree) {
  // Tree rooted at 0: 0->1, 0->2, 1->3, 2->4
  std::vector<std::pair<int, int>> edges = {{0, 1}, {0, 2}, {1, 3}, {2, 4}};
  auto order = teaser::topologicalSort(5, edges);

  ASSERT_EQ(order.size(), 5u);
  std::vector<int> pos(5);
  for (int i = 0; i < 5; ++i) {
    pos[order[i]] = i;
  }
  for (const auto& e : edges) {
    EXPECT_LT(pos[e.first], pos[e.second]); // parent before child
  }
  EXPECT_EQ(order.front(), 0); // root has no parents -> emitted first (smallest ready index)
}

// A vertex with multiple parents appears only after all of them.
TEST(MultiviewTest, TopologicalSortMultipleParents) {
  // Diamond: 0->1, 0->2, 1->3, 2->3  (node 3 has two parents)
  std::vector<std::pair<int, int>> edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
  auto order = teaser::topologicalSort(4, edges);

  ASSERT_EQ(order.size(), 4u);
  std::vector<int> pos(4);
  for (int i = 0; i < 4; ++i) {
    pos[order[i]] = i;
  }
  for (const auto& e : edges) {
    EXPECT_LT(pos[e.first], pos[e.second]);
  }
  EXPECT_GT(pos[3], pos[1]);
  EXPECT_GT(pos[3], pos[2]);
}

// A cycle has no valid ordering => empty result.
TEST(MultiviewTest, TopologicalSortCycleReturnsEmpty) {
  std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
  auto order = teaser::topologicalSort(3, edges);
  EXPECT_TRUE(order.empty());
}

// Full multi-scan alignment on a connected 4-scan graph: recover every global pose (relative to
// the auto-selected anchor) despite gross outliers on every edge.
TEST(MultiviewTest, MultiScanConnected) {
  std::vector<teaser::Pose> gt(4);
  gt[0].R = makeRotation(1, 0, 0, 0.0);        gt[0].t = Eigen::Vector3d(0, 0, 0);
  gt[1].R = makeRotation(0.2, -0.5, 1.0, 0.6); gt[1].t = Eigen::Vector3d(1, -2, 0.5);
  gt[2].R = makeRotation(0.7, 0.1, -0.3, 1.2); gt[2].t = Eigen::Vector3d(-1, 0.4, -1.5);
  gt[3].R = makeRotation(-1, 0.3, 0.4, -0.9);  gt[3].t = Eigen::Vector3d(-0.5, 1.5, 2.0);

  // node 2 has the most incident edges => it becomes the anchor.
  std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {0, 2}};
  auto scene = buildScene(gt, edges, /*k_in=*/30, /*k_out=*/8);

  auto res = teaser::alignMultiScan(scene.clouds, scene.adjacency, scene.edge_weights, scene.corr,
                                    makeParams(1e-3));

  ASSERT_EQ(res.num_components, 1);
  const int anchor = 2;
  EXPECT_TRUE(res.poses[anchor].R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
  EXPECT_LT(res.poses[anchor].t.norm(), 1e-9);
  EXPECT_LT(res.residual_to_parent[anchor], 0.0); // anchor has no parent

  for (int n = 0; n < 4; ++n) {
    EXPECT_TRUE(res.valid[n]);
    // GT pose expressed relative to the anchor (which is identity in the result).
    Eigen::Matrix3d R_rel = gt[anchor].R.transpose() * gt[n].R;
    Eigen::Vector3d t_rel = gt[anchor].R.transpose() * (gt[n].t - gt[anchor].t);
    EXPECT_LE(teaser::test::getAngularError(R_rel, res.poses[n].R), 1e-2);
    EXPECT_LE((res.poses[n].t - t_rel).norm(), 1e-2);
    if (n != anchor) {
      // Non-anchor nodes report a small, non-negative mean residual to their tree-parent.
      EXPECT_GE(res.residual_to_parent[n], 0.0);
      EXPECT_LT(res.residual_to_parent[n], 1e-2);
    }
  }
}

// Disconnected graph: two components aligned independently, each with its own identity anchor.
TEST(MultiviewTest, MultiScanDisconnected) {
  std::vector<teaser::Pose> gt(4);
  gt[0].R = makeRotation(1, 0, 0, 0.0);        gt[0].t = Eigen::Vector3d(0, 0, 0);
  gt[1].R = makeRotation(0.2, -0.5, 1.0, 0.6); gt[1].t = Eigen::Vector3d(1, -2, 0.5);
  gt[2].R = makeRotation(0.7, 0.1, -0.3, 1.2); gt[2].t = Eigen::Vector3d(-1, 0.4, -1.5);
  gt[3].R = makeRotation(-1, 0.3, 0.4, -0.9);  gt[3].t = Eigen::Vector3d(-0.5, 1.5, 2.0);

  // Two disjoint components: {0,1} and {2,3}.
  std::vector<std::pair<int, int>> edges = {{0, 1}, {2, 3}};
  auto scene = buildScene(gt, edges, /*k_in=*/30, /*k_out=*/5);

  auto res = teaser::alignMultiScan(scene.clouds, scene.adjacency, scene.edge_weights, scene.corr,
                                    makeParams(1e-3));

  EXPECT_EQ(res.num_components, 2);
  EXPECT_EQ(res.component[0], res.component[1]);
  EXPECT_EQ(res.component[2], res.component[3]);
  EXPECT_NE(res.component[0], res.component[2]);

  // Each component's anchor (tie => smallest index) is identity: nodes 0 and 2.
  EXPECT_TRUE(res.poses[0].R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
  EXPECT_TRUE(res.poses[2].R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));

  // Within each component, the non-anchor pose matches GT relative to its anchor.
  auto check_rel = [&](int anchor, int n) {
    Eigen::Matrix3d R_rel = gt[anchor].R.transpose() * gt[n].R;
    Eigen::Vector3d t_rel = gt[anchor].R.transpose() * (gt[n].t - gt[anchor].t);
    EXPECT_TRUE(res.valid[n]);
    EXPECT_LE(teaser::test::getAngularError(R_rel, res.poses[n].R), 1e-2);
    EXPECT_LE((res.poses[n].t - t_rel).norm(), 1e-2);
  };
  check_rel(0, 1);
  check_rel(2, 3);
}

// Caller-provided tree (MST step skipped): same recovery as the MST path.
TEST(MultiviewTest, MultiScanWithCallerTree) {
  std::vector<teaser::Pose> gt(4);
  gt[0].R = makeRotation(1, 0, 0, 0.0);        gt[0].t = Eigen::Vector3d(0, 0, 0);
  gt[1].R = makeRotation(0.2, -0.5, 1.0, 0.6); gt[1].t = Eigen::Vector3d(1, -2, 0.5);
  gt[2].R = makeRotation(0.7, 0.1, -0.3, 1.2); gt[2].t = Eigen::Vector3d(-1, 0.4, -1.5);
  gt[3].R = makeRotation(-1, 0.3, 0.4, -0.9);  gt[3].t = Eigen::Vector3d(-0.5, 1.5, 2.0);

  std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {0, 2}};
  auto scene = buildScene(gt, edges, /*k_in=*/30, /*k_out=*/8);

  // Caller supplies its own tree: a star centered at node 2 (skips the internal MST).
  std::vector<std::pair<int, int>> tree = {{0, 2}, {1, 2}, {2, 3}};
  auto res = teaser::alignMultiScanWithTree(scene.clouds, tree, scene.corr, makeParams(1e-3));

  ASSERT_EQ(res.num_components, 1);
  const int anchor = 2; // highest degree in the provided tree
  EXPECT_TRUE(res.poses[anchor].R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
  EXPECT_LT(res.poses[anchor].t.norm(), 1e-9);

  for (int n = 0; n < 4; ++n) {
    EXPECT_TRUE(res.valid[n]);
    Eigen::Matrix3d R_rel = gt[anchor].R.transpose() * gt[n].R;
    Eigen::Vector3d t_rel = gt[anchor].R.transpose() * (gt[n].t - gt[anchor].t);
    EXPECT_LE(teaser::test::getAngularError(R_rel, res.poses[n].R), 1e-2);
    EXPECT_LE((res.poses[n].t - t_rel).norm(), 1e-2);
  }
}

// Degenerate input (too few correspondences) is reported as invalid.
TEST(MultiviewTest, DegenerateInput) {
  teaser::MultiviewSolver solver(makeParams(1e-4));

  // No edges at all.
  auto empty = solver.solveNodePose({});
  EXPECT_FALSE(empty.valid);

  // A single correspondence (< 3 total).
  teaser::NeighborEdge edge;
  edge.R_i = Eigen::Matrix3d::Identity();
  edge.t_i = Eigen::Vector3d::Zero();
  edge.src = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, 1);
  edge.dst = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, 1);
  auto too_few = solver.solveNodePose({edge});
  EXPECT_FALSE(too_few.valid);
}
