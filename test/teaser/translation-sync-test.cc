/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "translation_sync.h" // private header under teaser/src

namespace {

using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;
using Pts = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Wts = Eigen::Matrix<double, 1, Eigen::Dynamic>;

// Deterministic RNG so the synthetic geometry is reproducible across runs.
std::mt19937& rng() {
  static std::mt19937 gen(5150);
  return gen;
}

Mat3 randomRotation() {
  std::uniform_real_distribution<double> axis(-1.0, 1.0);
  std::uniform_real_distribution<double> angle(-M_PI, M_PI);
  Eigen::Vector3d a(axis(rng()), axis(rng()), axis(rng()));
  if (a.norm() < 1e-6) {
    a = Eigen::Vector3d::UnitZ();
  }
  return Eigen::AngleAxisd(angle(rng()), a.normalized()).toRotationMatrix();
}

Vec3 randomTranslation() {
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  return Vec3(dist(rng()), dist(rng()), dist(rng()));
}

Pts sampleWorldPoints(int m) {
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  Pts W(3, m);
  for (int i = 0; i < m; ++i) {
    W.col(i) << dist(rng()), dist(rng()), dist(rng());
  }
  return W;
}

/**
 * Build edge (p,q) from ground-truth poses. Each scan sees the same world points in its own local
 * frame, p_pts_j = R_p^T (W_j - t_p), so R_p p_pts_j + t_p == R_q q_pts_j + t_q == W_j.
 * `noise` perturbs q's side.
 */
teaser::TranslationSyncEdge makeEdge(int p, int q, const std::vector<Mat3>& R_gt,
                                     const std::vector<Vec3>& t_gt, int m, double noise = 0.0) {
  const Pts W = sampleWorldPoints(m);
  teaser::TranslationSyncEdge e;
  e.p = p;
  e.q = q;
  e.p_pts = R_gt[p].transpose() * (W.colwise() - t_gt[p]);
  e.q_pts = R_gt[q].transpose() * (W.colwise() - t_gt[q]);
  if (noise > 0) {
    std::normal_distribution<double> g(0.0, noise);
    for (int j = 0; j < m; ++j) {
      e.q_pts.col(j) += Vec3(g(rng()), g(rng()), g(rng()));
    }
  }
  e.w = Wts::Ones(1, m);
  return e;
}

/** The objective: sum over edges of sum_j w_j ||(R_p p_j + t_p) - (R_q q_j + t_q)||^2. */
double syncCost(const std::vector<teaser::TranslationSyncEdge>& edges,
                const std::vector<Mat3>& R, const std::vector<Vec3>& t) {
  double cost = 0;
  for (const auto& e : edges) {
    const Pts lhs = (R[e.p] * e.p_pts).colwise() + t[e.p];
    const Pts rhs = (R[e.q] * e.q_pts).colwise() + t[e.q];
    const Pts diff = lhs - rhs;
    cost += (diff.colwise().squaredNorm().array() * e.w.array()).sum();
  }
  return cost;
}

void expectVecNear(const Vec3& actual, const Vec3& expected, double tol) {
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(actual(i), expected(i), tol) << "at component " << i;
  }
}

/**
 * The gauge is arbitrary, so compare against the ground truth re-expressed in the gauge the solver
 * pins: the lowest-indexed node of the component has zero translation.
 */
void expectMatchesGroundTruth(const teaser::TranslationSyncResult& res,
                              const std::vector<Vec3>& t_gt, const std::vector<int>& nodes,
                              double tol) {
  const Vec3 anchor = t_gt[nodes.front()];
  for (int p : nodes) {
    ASSERT_TRUE(res.valid[p]) << "node " << p;
    expectVecNear(res.translations[p], t_gt[p] - anchor, tol);
  }
}

/** Ground truth with node 0's translation subtracted, matching the solver's gauge. */
std::vector<Vec3> anchored(const std::vector<Vec3>& t_gt) {
  std::vector<Vec3> out;
  out.reserve(t_gt.size());
  for (const auto& t : t_gt) {
    out.push_back(t - t_gt.front());
  }
  return out;
}

} // namespace

// Noise-free recovery on a fully connected triangle. Note the rotations are NOT identity and the
// translations are large, so a sign or transpose slip in the target d = R_q qbar - R_p pbar shows
// up immediately.
TEST(TranslationSyncTest, RecoversTriangleExactly) {
  for (int trial = 0; trial < 10; ++trial) {
    std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
    std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation()};
    std::vector<teaser::TranslationSyncEdge> edges = {makeEdge(0, 1, R_gt, t_gt, 40),
                                                      makeEdge(1, 2, R_gt, t_gt, 40),
                                                      makeEdge(0, 2, R_gt, t_gt, 40)};

    const auto res = teaser::synchronizeTranslations(3, edges, R_gt);

    EXPECT_EQ(res.num_components, 1);
    expectMatchesGroundTruth(res, t_gt, {0, 1, 2}, 1e-9);
    // The anchor is pinned to exactly zero.
    expectVecNear(res.translations[0], Vec3::Zero(), 1e-15);
    EXPECT_NEAR(syncCost(edges, R_gt, res.translations), 0.0, 1e-16);
  }
}

// The per-edge reduction to a single weighted centroid is claimed to be exact, not an
// approximation. Verify against the true minimizer: the solution must beat any perturbation of it.
TEST(TranslationSyncTest, ReductionIsExactOnNoisyData) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(),
                            randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation(),
                            randomTranslation()};
  // Over-determined: 4-cycle plus both diagonals, so no exact solution exists.
  std::vector<teaser::TranslationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, t_gt, 20, 0.05), makeEdge(1, 2, R_gt, t_gt, 20, 0.05),
      makeEdge(2, 3, R_gt, t_gt, 20, 0.05), makeEdge(0, 3, R_gt, t_gt, 20, 0.05),
      makeEdge(0, 2, R_gt, t_gt, 20, 0.05), makeEdge(1, 3, R_gt, t_gt, 20, 0.05)};

  const auto res = teaser::synchronizeTranslations(4, edges, R_gt);
  ASSERT_EQ(res.num_components, 1);
  const double best = syncCost(edges, R_gt, res.translations);

  // Any perturbation that is not a pure global shift must be worse. Perturbing a single node is
  // never a global shift.
  std::normal_distribution<double> g(0.0, 0.02);
  for (int trial = 0; trial < 30; ++trial) {
    std::vector<Vec3> perturbed = res.translations;
    const int node = 1 + (trial % 3); // never the anchor, which is fixed by the gauge
    perturbed[node] += Vec3(g(rng()), g(rng()), g(rng()));
    EXPECT_GT(syncCost(edges, R_gt, perturbed), best) << "trial " << trial;
  }
}

// Zero-weight correspondences are the GNC line process at work: they must be exactly as if never
// supplied, however wrong their geometry.
TEST(TranslationSyncTest, ZeroWeightCorrespondencesAreIgnored) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation()};
  std::vector<teaser::TranslationSyncEdge> clean = {makeEdge(0, 1, R_gt, t_gt, 20),
                                                    makeEdge(1, 2, R_gt, t_gt, 20),
                                                    makeEdge(0, 2, R_gt, t_gt, 20)};

  std::vector<teaser::TranslationSyncEdge> padded = clean;
  std::uniform_real_distribution<double> junk(-50.0, 50.0);
  for (auto& e : padded) {
    const Eigen::Index m = e.p_pts.cols();
    e.p_pts.conservativeResize(3, m + 5);
    e.q_pts.conservativeResize(3, m + 5);
    e.w.conservativeResize(1, m + 5);
    for (Eigen::Index j = m; j < m + 5; ++j) {
      e.p_pts.col(j) << junk(rng()), junk(rng()), junk(rng());
      e.q_pts.col(j) << junk(rng()), junk(rng()), junk(rng());
      e.w(j) = 0.0;
    }
  }

  const auto clean_res = teaser::synchronizeTranslations(3, clean, R_gt);
  const auto padded_res = teaser::synchronizeTranslations(3, padded, R_gt);

  expectMatchesGroundTruth(padded_res, t_gt, {0, 1, 2}, 1e-9);
  for (int p = 0; p < 3; ++p) {
    expectVecNear(padded_res.translations[p], clean_res.translations[p], 1e-9);
  }
}

// Recentering the clouds must not change the anchored answer -- the whole reason Step 0 does not do
// it. Shift each scan's local points by an arbitrary offset and undo it the documented way.
TEST(TranslationSyncTest, RecenteringIsANoOp) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(),
                            randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation(),
                            randomTranslation()};
  std::vector<teaser::TranslationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, t_gt, 25, 0.02), makeEdge(1, 2, R_gt, t_gt, 25, 0.02),
      makeEdge(2, 3, R_gt, t_gt, 25, 0.02), makeEdge(0, 3, R_gt, t_gt, 25, 0.02)};

  const auto base = teaser::synchronizeTranslations(4, edges, R_gt);

  // Arbitrary per-node offsets, deliberately large.
  const std::vector<Vec3> o = {Vec3(10, -3, 7), Vec3(-40, 12, 0.5), Vec3(0, 0, 100),
                               Vec3(2.5, 2.5, -2.5)};
  std::vector<teaser::TranslationSyncEdge> shifted = edges;
  for (auto& e : shifted) {
    e.p_pts.colwise() -= o[e.p];
    e.q_pts.colwise() -= o[e.q];
  }
  const auto res = teaser::synchronizeTranslations(4, shifted, R_gt);
  ASSERT_EQ(res.num_components, 1);

  // Undo: t_p -= R_p o_p, then re-anchor since the gauge is pinned in the shifted variables.
  std::vector<Vec3> undone(4);
  for (int p = 0; p < 4; ++p) {
    undone[p] = res.translations[p] - R_gt[p] * o[p];
  }
  for (int p = 3; p >= 0; --p) {
    undone[p] -= undone[0];
  }
  for (int p = 0; p < 4; ++p) {
    expectVecNear(undone[p], base.translations[p], 1e-9);
  }
}

// Uniform noise bounds scale every edge weight by a common factor, which cancels out of the
// Laplacian system; differing bounds must actually move the answer.
TEST(TranslationSyncTest, UniformNoiseBoundsAreANoOpAndDifferingOnesAreNot) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(),
                            randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation(),
                            randomTranslation()};
  std::vector<teaser::TranslationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, t_gt, 25, 0.02), makeEdge(1, 2, R_gt, t_gt, 25, 0.02),
      makeEdge(2, 3, R_gt, t_gt, 25, 0.02), makeEdge(0, 3, R_gt, t_gt, 25, 0.02)};

  const auto base = teaser::synchronizeTranslations(4, edges, R_gt);

  for (double delta : {1e-4, 0.05, 12.0}) {
    const std::vector<double> bounds(4, delta);
    const auto res = teaser::synchronizeTranslations(4, edges, R_gt,
                                                     teaser::TranslationSyncParams(), nullptr,
                                                     &bounds);
    for (int p = 0; p < 4; ++p) {
      expectVecNear(res.translations[p], base.translations[p], 1e-9);
    }
  }

  const std::vector<double> mixed = {0.01, 0.04, 0.20, 0.02};
  const auto weighted = teaser::synchronizeTranslations(
      4, edges, R_gt, teaser::TranslationSyncParams(), nullptr, &mixed);
  double max_diff = 0;
  for (int p = 0; p < 4; ++p) {
    max_diff = std::max(max_diff, (weighted.translations[p] - base.translations[p]).norm());
  }
  EXPECT_GT(max_diff, 1e-6);
}

// No factor of 2 here, unlike the rotation stage's timResidualNoiseBound.
TEST(TranslationSyncTest, PointResidualNoiseBoundComposition) {
  EXPECT_DOUBLE_EQ(teaser::pointResidualNoiseBound(0.01, 0.01), 0.02);
  EXPECT_DOUBLE_EQ(teaser::pointResidualNoiseBound(0.01, 0.05), 0.06);
  EXPECT_DOUBLE_EQ(teaser::pointResidualNoiseBound(0.03, 0.07),
                   teaser::pointResidualNoiseBound(0.07, 0.03));
  EXPECT_DOUBLE_EQ(teaser::pointResidualNoiseBound(0.0, 0.02), 0.02);
}

// A scan known to be noisy should pull less on its neighbors.
TEST(TranslationSyncTest, NoisyScanIsDownweighted) {
  constexpr int kTrials = 20;
  constexpr double kClean = 0.005;
  constexpr double kDirty = 0.20;
  const std::vector<double> bounds = {kClean, kClean, kDirty, kClean};

  double weighted_err = 0;
  double unweighted_err = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(),
                              randomRotation()};
    std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation(),
                              randomTranslation()};
    std::vector<teaser::TranslationSyncEdge> edges = {
        makeEdge(0, 1, R_gt, t_gt, 25, kClean), makeEdge(1, 2, R_gt, t_gt, 25, kDirty),
        makeEdge(2, 3, R_gt, t_gt, 25, kDirty), makeEdge(0, 3, R_gt, t_gt, 25, kClean)};

    const auto plain = teaser::synchronizeTranslations(4, edges, R_gt);
    const auto weighted = teaser::synchronizeTranslations(
        4, edges, R_gt, teaser::TranslationSyncParams(), nullptr, &bounds);

    const std::vector<Vec3> truth = anchored(t_gt);
    for (int p : {1, 3}) {
      unweighted_err += (plain.translations[p] - truth[p]).norm();
      weighted_err += (weighted.translations[p] - truth[p]).norm();
    }
  }

  EXPECT_LT(weighted_err, unweighted_err)
      << "weighted=" << weighted_err << " unweighted=" << unweighted_err;
}

// Components split, are gauged independently, and share the rotation stage's convention: the
// lowest-indexed node of each is the anchor.
TEST(TranslationSyncTest, DisconnectedComponentsAreGaugedIndependently) {
  std::vector<Mat3> R_gt;
  std::vector<Vec3> t_gt;
  for (int i = 0; i < 6; ++i) {
    R_gt.push_back(randomRotation());
    t_gt.push_back(randomTranslation());
  }
  std::vector<teaser::TranslationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, t_gt, 20), makeEdge(1, 2, R_gt, t_gt, 20),
      makeEdge(0, 2, R_gt, t_gt, 20), makeEdge(3, 4, R_gt, t_gt, 20),
      makeEdge(4, 5, R_gt, t_gt, 20), makeEdge(3, 5, R_gt, t_gt, 20)};

  const auto res = teaser::synchronizeTranslations(6, edges, R_gt);

  EXPECT_EQ(res.num_components, 2);
  EXPECT_EQ(res.component[0], res.component[2]);
  EXPECT_NE(res.component[0], res.component[3]);
  expectMatchesGroundTruth(res, t_gt, {0, 1, 2}, 1e-9);
  expectMatchesGroundTruth(res, t_gt, {3, 4, 5}, 1e-9);
  expectVecNear(res.translations[0], Vec3::Zero(), 1e-15);
  expectVecNear(res.translations[3], Vec3::Zero(), 1e-15);
}

TEST(TranslationSyncTest, UnconstrainedNodeFallsBackToInitialGuess) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation()};
  std::vector<teaser::TranslationSyncEdge> edges = {makeEdge(0, 1, R_gt, t_gt, 20)};

  const auto plain = teaser::synchronizeTranslations(3, edges, R_gt);
  EXPECT_FALSE(plain.valid[2]);
  EXPECT_EQ(plain.component[2], -1);
  expectVecNear(plain.translations[2], Vec3::Zero(), 1e-15);

  const Vec3 previous = randomTranslation();
  const std::vector<Vec3> initial = {Vec3::Zero(), Vec3::Zero(), previous};
  const auto warm = teaser::synchronizeTranslations(3, edges, R_gt,
                                                    teaser::TranslationSyncParams(), &initial);
  EXPECT_FALSE(warm.valid[2]);
  expectVecNear(warm.translations[2], previous, 1e-15);
  expectMatchesGroundTruth(warm, t_gt, {0, 1}, 1e-9);
}

TEST(TranslationSyncTest, FullyDownweightedEdgeIsDropped) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation()};
  std::vector<teaser::TranslationSyncEdge> edges = {makeEdge(0, 1, R_gt, t_gt, 20),
                                                    makeEdge(1, 2, R_gt, t_gt, 20)};
  edges[1].w.setZero(); // node 2 loses its only edge

  const auto res = teaser::synchronizeTranslations(3, edges, R_gt);

  EXPECT_EQ(res.num_components, 1);
  EXPECT_TRUE(res.valid[1]);
  EXPECT_FALSE(res.valid[2]);
  EXPECT_EQ(res.component[2], -1);
  expectMatchesGroundTruth(res, t_gt, {0, 1}, 1e-9);
}

TEST(TranslationSyncTest, HandlesDegenerateInputs) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation()};

  // Smallest real problem: two nodes, one edge.
  std::vector<teaser::TranslationSyncEdge> one = {makeEdge(0, 1, R_gt, t_gt, 15)};
  const auto res = teaser::synchronizeTranslations(2, one, R_gt);
  EXPECT_EQ(res.num_components, 1);
  expectMatchesGroundTruth(res, t_gt, {0, 1}, 1e-9);

  // No edges, no nodes.
  const std::vector<Mat3> R3(3, Mat3::Identity());
  const auto empty = teaser::synchronizeTranslations(3, {}, R3);
  EXPECT_EQ(empty.num_components, 0);
  EXPECT_EQ(empty.translations.size(), 3u);
  const auto none = teaser::synchronizeTranslations(0, {}, {});
  EXPECT_TRUE(none.translations.empty());

  // A rotations vector of the wrong size is refused outright rather than read out of bounds.
  const auto bad_rot = teaser::synchronizeTranslations(2, one, {Mat3::Identity()});
  EXPECT_TRUE(bad_rot.translations.empty());

  // Malformed edges are skipped: out of range, self loop, mismatched column counts.
  teaser::TranslationSyncEdge bad_dims = makeEdge(0, 1, R_gt, t_gt, 10);
  bad_dims.w = Wts::Ones(1, 4);
  std::vector<teaser::TranslationSyncEdge> bad = {makeEdge(0, 1, R_gt, t_gt, 10), bad_dims};
  bad[0].q = 5;    // out of range
  bad.push_back(makeEdge(0, 1, R_gt, t_gt, 10));
  bad[2].q = 0;    // self loop
  const auto skipped = teaser::synchronizeTranslations(2, bad, R_gt);
  EXPECT_EQ(skipped.num_components, 0);
}

// Diagnostics are off by default and populated on request.
TEST(TranslationSyncTest, Diagnostics) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(),
                            randomRotation()};
  std::vector<Vec3> t_gt = {randomTranslation(), randomTranslation(), randomTranslation(),
                            randomTranslation()};
  // Over-determined so there is redundancy to estimate a residual scale from.
  std::vector<teaser::TranslationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, t_gt, 25, 0.02), makeEdge(1, 2, R_gt, t_gt, 25, 0.02),
      makeEdge(2, 3, R_gt, t_gt, 25, 0.02), makeEdge(0, 3, R_gt, t_gt, 25, 0.02),
      makeEdge(0, 2, R_gt, t_gt, 25, 0.02)};

  const auto off = teaser::synchronizeTranslations(4, edges, R_gt);
  EXPECT_LT(off.residual_sigma[0], 0.0);
  EXPECT_LT(off.fiedler_value[0], 0.0);

  teaser::TranslationSyncParams params;
  params.compute_diagnostics = true;
  const std::vector<std::pair<int, int>> pairs = {{0, 2}, {1, 3}};
  const auto on = teaser::synchronizeTranslations(4, edges, R_gt, params, nullptr, nullptr, &pairs);

  EXPECT_GT(on.residual_sigma[0], 0.0);
  EXPECT_GT(on.fiedler_value[0], 0.0); // connected => lambda_2 > 0
  EXPECT_EQ(on.pair_sigma.size(), 2u);
  for (const auto& kv : on.pair_sigma) {
    EXPECT_GT(kv.second, 0.0);
  }
  // Solving does not depend on the diagnostics being on.
  for (int p = 0; p < 4; ++p) {
    expectVecNear(on.translations[p], off.translations[p], 1e-12);
  }

  // A spanning tree has no redundancy, so there is no residual scale to report.
  std::vector<teaser::TranslationSyncEdge> tree = {makeEdge(0, 1, R_gt, t_gt, 25, 0.02),
                                                   makeEdge(1, 2, R_gt, t_gt, 25, 0.02),
                                                   makeEdge(2, 3, R_gt, t_gt, 25, 0.02)};
  const auto tree_res =
      teaser::synchronizeTranslations(4, tree, R_gt, params, nullptr, nullptr, &pairs);
  EXPECT_LT(tree_res.residual_sigma[0], 0.0);
  EXPECT_TRUE(tree_res.pair_sigma.empty());
  // A tree is exactly determined, so it still fits its edges perfectly.
  EXPECT_GT(tree_res.fiedler_value[0], 0.0);
}

// A chain is weakly connected compared to a fully connected graph; the Fiedler value must say so.
TEST(TranslationSyncTest, FiedlerValueReflectsConnectivity) {
  std::vector<Mat3> R_gt;
  std::vector<Vec3> t_gt;
  for (int i = 0; i < 5; ++i) {
    R_gt.push_back(randomRotation());
    t_gt.push_back(randomTranslation());
  }
  teaser::TranslationSyncParams params;
  params.compute_diagnostics = true;

  std::vector<teaser::TranslationSyncEdge> chain;
  for (int i = 0; i + 1 < 5; ++i) {
    chain.push_back(makeEdge(i, i + 1, R_gt, t_gt, 20));
  }
  std::vector<teaser::TranslationSyncEdge> complete = chain;
  for (int i = 0; i < 5; ++i) {
    for (int j = i + 2; j < 5; ++j) {
      complete.push_back(makeEdge(i, j, R_gt, t_gt, 20));
    }
  }

  const auto chain_res = teaser::synchronizeTranslations(5, chain, R_gt, params);
  const auto complete_res = teaser::synchronizeTranslations(5, complete, R_gt, params);

  expectMatchesGroundTruth(chain_res, t_gt, {0, 1, 2, 3, 4}, 1e-9);
  expectMatchesGroundTruth(complete_res, t_gt, {0, 1, 2, 3, 4}, 1e-9);
  EXPECT_GT(complete_res.fiedler_value[0], chain_res.fiedler_value[0]);
}
