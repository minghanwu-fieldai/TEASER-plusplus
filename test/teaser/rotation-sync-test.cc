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

#include "rotation_sync.h" // private header under teaser/src
#include "teaser/utils.h"

namespace {

using Mat3 = Eigen::Matrix3d;
using Pts = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Wts = Eigen::Matrix<double, 1, Eigen::Dynamic>;

// Deterministic RNG so the synthetic geometry is reproducible across runs.
std::mt19937& rng() {
  static std::mt19937 gen(7331);
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

Pts sampleWorldPoints(int m) {
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  Pts W(3, m);
  for (int i = 0; i < m; ++i) {
    W.col(i) << dist(rng()), dist(rng()), dist(rng());
  }
  return W;
}

// Arbitrary but distinct per-node translations. TIMs are differences, so these must cancel exactly;
// having them nonzero is what makes every test below a check on translation invariance.
Eigen::Vector3d nodeTranslation(int i) {
  return Eigen::Vector3d(3.0 * i + 1.0, -2.0 * i - 0.5, 0.75 * i + 4.0);
}

/**
 * Build edge (p,q) from ground-truth poses, as genuine TIMs rather than raw points.
 *
 * Each scan sees the world points in its own local frame, a_j = R_p^T (W_j - t_p), and the TIMs are
 * consecutive differences of those. The translation drops out of the difference, leaving
 * p_tims_j = R_p^T d_j and q_tims_j = R_q^T d_j for the same world displacement d_j, so at the
 * optimum R_p * p_tims_j == R_q * q_tims_j == d_j. `noise` perturbs q's side.
 */
teaser::RotationSyncEdge makeEdge(int p, int q, const std::vector<Mat3>& R_gt, int m,
                                  double noise = 0.0) {
  // m + 1 points give m consecutive differences.
  const Pts W = sampleWorldPoints(m + 1);
  const Pts a = R_gt[p].transpose() * (W.colwise() - nodeTranslation(p));
  const Pts b = R_gt[q].transpose() * (W.colwise() - nodeTranslation(q));

  teaser::RotationSyncEdge e;
  e.p = p;
  e.q = q;
  e.p_tims = a.rightCols(m) - a.leftCols(m);
  e.q_tims = b.rightCols(m) - b.leftCols(m);
  if (noise > 0) {
    std::normal_distribution<double> g(0.0, noise);
    for (int j = 0; j < m; ++j) {
      e.q_tims.col(j) += Eigen::Vector3d(g(rng()), g(rng()), g(rng()));
    }
  }
  e.w = Wts::Ones(1, m);
  return e;
}

/** The objective being relaxed: sum over edges of sum_j w_j ||R_p x_j - R_q y_j||^2. */
double syncCost(const std::vector<teaser::RotationSyncEdge>& edges, const std::vector<Mat3>& R) {
  double cost = 0;
  for (const auto& e : edges) {
    const Pts diff = R[e.p] * e.p_tims - R[e.q] * e.q_tims;
    cost += (diff.colwise().squaredNorm().array() * e.w.array()).sum();
  }
  return cost;
}

void expectRotationNear(const Mat3& actual, const Mat3& expected, double tol) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(actual(i, j), expected(i, j), tol) << "at (" << i << ", " << j << ")";
    }
  }
}

void expectProperRotation(const Mat3& R, double tol = 1e-9) {
  EXPECT_NEAR(R.determinant(), 1.0, tol);
  expectRotationNear(R * R.transpose(), Mat3::Identity(), tol);
}

/**
 * The gauge is arbitrary, so compare every node against the ground truth re-expressed in the same
 * gauge the solver pins: the lowest-indexed node of the component is the identity.
 */
void expectMatchesGroundTruth(const teaser::RotationSyncResult& res, const std::vector<Mat3>& R_gt,
                              const std::vector<int>& nodes, double tol) {
  const Mat3 anchor_inv = R_gt[nodes.front()].transpose();
  for (int p : nodes) {
    ASSERT_TRUE(res.valid[p]) << "node " << p;
    EXPECT_TRUE(res.reliable[p]) << "node " << p;
    expectProperRotation(res.rotations[p]);
    expectRotationNear(res.rotations[p], anchor_inv * R_gt[p], tol);
  }
}

} // namespace

// Noise-free recovery on a fully connected triangle. This is the test that pins the Rhat
// convention: a transposed relative rotation still yields perfectly valid rotation matrices, so
// only a ground-truth comparison detects it. Repeated over several draws so both parities of the
// eigenbasis gauge (det Q = +1 and -1) get exercised.
TEST(RotationSyncTest, RecoversTriangleExactly) {
  for (int trial = 0; trial < 10; ++trial) {
    std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
    std::vector<teaser::RotationSyncEdge> edges = {
        makeEdge(0, 1, R_gt, 40), makeEdge(1, 2, R_gt, 40), makeEdge(0, 2, R_gt, 40)};

    const auto res = teaser::synchronizeRotations(3, edges);

    EXPECT_EQ(res.num_components, 1);
    expectMatchesGroundTruth(res, R_gt, {0, 1, 2}, 1e-9);
    // The anchor is pinned to exactly the identity.
    expectRotationNear(res.rotations[0], Mat3::Identity(), 1e-12);
    EXPECT_NEAR(syncCost(edges, res.rotations), 0.0, 1e-16);
  }
}

// Both a path and a cycle recover exactly without noise, but the extra edge makes the top-3
// eigenspace better separated.
TEST(RotationSyncTest, CycleHasLargerSpectralGapThanChain) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(), randomRotation()};

  std::vector<teaser::RotationSyncEdge> chain = {
      makeEdge(0, 1, R_gt, 30), makeEdge(1, 2, R_gt, 30), makeEdge(2, 3, R_gt, 30)};
  std::vector<teaser::RotationSyncEdge> cycle = chain;
  cycle.push_back(makeEdge(0, 3, R_gt, 30));

  const auto chain_res = teaser::synchronizeRotations(4, chain);
  const auto cycle_res = teaser::synchronizeRotations(4, cycle);

  expectMatchesGroundTruth(chain_res, R_gt, {0, 1, 2, 3}, 1e-9);
  expectMatchesGroundTruth(cycle_res, R_gt, {0, 1, 2, 3}, 1e-9);

  ASSERT_EQ(chain_res.spectral_gap.size(), 1u);
  ASSERT_EQ(cycle_res.spectral_gap.size(), 1u);
  EXPECT_GT(chain_res.spectral_gap[0], 0.0);
  EXPECT_GT(cycle_res.spectral_gap[0], chain_res.spectral_gap[0]);
}

// The reason the helper exists: sequential propagation along a chain accumulates drift and cannot
// absorb a loop closure, while joint synchronization distributes the error over all edges.
TEST(RotationSyncTest, JointSolveBeatsSequentialPropagation) {
  constexpr int kTrials = 20;
  double sync_total = 0;
  double chain_total = 0;
  int sync_wins = 0;

  for (int trial = 0; trial < kTrials; ++trial) {
    std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(),
                              randomRotation()};
    // 4-cycle: 0-1-2-3-0. The last edge is the loop closure.
    std::vector<teaser::RotationSyncEdge> edges = {
        makeEdge(0, 1, R_gt, 25, 0.05), makeEdge(1, 2, R_gt, 25, 0.05),
        makeEdge(2, 3, R_gt, 25, 0.05), makeEdge(0, 3, R_gt, 25, 0.05)};

    // Sequential baseline: anchor node 0, then walk the chain with pairwise svdRot, ignoring the
    // closure edge during estimation (exactly what a spanning-tree propagation does).
    std::vector<Mat3> chain_R(4, Mat3::Identity());
    for (int k = 0; k < 3; ++k) {
      // Rhat == R_q^T R_p, so R_q = R_p * Rhat^T.
      const Mat3 Rhat = teaser::utils::svdRot(edges[k].p_tims, edges[k].q_tims, edges[k].w);
      chain_R[edges[k].q] = chain_R[edges[k].p] * Rhat.transpose();
    }

    const auto res = teaser::synchronizeRotations(4, edges);
    ASSERT_EQ(res.num_components, 1);

    // Both are evaluated on the full edge set, closure included, in a common gauge.
    const double c_sync = syncCost(edges, res.rotations);
    const double c_chain = syncCost(edges, chain_R);
    sync_total += c_sync;
    chain_total += c_chain;
    if (c_sync < c_chain) {
      ++sync_wins;
    }
  }

  EXPECT_LT(sync_total, chain_total);
  EXPECT_GE(sync_wins, kTrials - 2) << "joint solve lost on too many trials";
}

// The per-correspondence weights are the GNC line process, so a zero-weight correspondence must be
// exactly as if it were never supplied -- however wrong its geometry is.
TEST(RotationSyncTest, ZeroWeightCorrespondencesAreIgnored) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<teaser::RotationSyncEdge> clean = {makeEdge(0, 1, R_gt, 20), makeEdge(1, 2, R_gt, 20),
                                                 makeEdge(0, 2, R_gt, 20)};

  // Same edges, but each padded with gross outliers carrying zero weight.
  std::vector<teaser::RotationSyncEdge> padded = clean;
  std::uniform_real_distribution<double> junk(-50.0, 50.0);
  for (auto& e : padded) {
    const Eigen::Index m = e.p_tims.cols();
    e.p_tims.conservativeResize(3, m + 5);
    e.q_tims.conservativeResize(3, m + 5);
    e.w.conservativeResize(1, m + 5);
    for (Eigen::Index j = m; j < m + 5; ++j) {
      e.p_tims.col(j) << junk(rng()), junk(rng()), junk(rng());
      e.q_tims.col(j) << junk(rng()), junk(rng()), junk(rng());
      e.w(j) = 0.0;
    }
  }

  const auto clean_res = teaser::synchronizeRotations(3, clean);
  const auto padded_res = teaser::synchronizeRotations(3, padded);

  expectMatchesGroundTruth(padded_res, R_gt, {0, 1, 2}, 1e-9);
  for (int p = 0; p < 3; ++p) {
    expectRotationNear(padded_res.rotations[p], clean_res.rotations[p], 1e-9);
  }
}

// An edge whose weights have all collapsed carries no constraint, so it must not hold a component
// together. Mid-GNC this is how a graph fragments.
TEST(RotationSyncTest, FullyDownweightedEdgeIsDropped) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<teaser::RotationSyncEdge> edges = {makeEdge(0, 1, R_gt, 20), makeEdge(1, 2, R_gt, 20)};
  edges[1].w.setZero(); // node 2 loses its only edge

  const auto res = teaser::synchronizeRotations(3, edges);

  EXPECT_EQ(res.num_components, 1);
  EXPECT_TRUE(res.valid[0]);
  EXPECT_TRUE(res.valid[1]);
  EXPECT_FALSE(res.valid[2]);
  EXPECT_EQ(res.component[2], -1);
  expectMatchesGroundTruth(res, R_gt, {0, 1}, 1e-9);
  expectRotationNear(res.rotations[2], Mat3::Identity(), 1e-12);
}

// Two triangles with no edge between them: each is synchronized independently and gets its own
// identity-anchored gauge.
TEST(RotationSyncTest, DisconnectedComponentsAreGaugedIndependently) {
  std::vector<Mat3> R_gt;
  for (int i = 0; i < 6; ++i) {
    R_gt.push_back(randomRotation());
  }
  std::vector<teaser::RotationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, 20), makeEdge(1, 2, R_gt, 20), makeEdge(0, 2, R_gt, 20),
      makeEdge(3, 4, R_gt, 20), makeEdge(4, 5, R_gt, 20), makeEdge(3, 5, R_gt, 20)};

  const auto res = teaser::synchronizeRotations(6, edges);

  EXPECT_EQ(res.num_components, 2);
  EXPECT_EQ(res.component[0], res.component[2]);
  EXPECT_EQ(res.component[3], res.component[5]);
  EXPECT_NE(res.component[0], res.component[3]);

  expectMatchesGroundTruth(res, R_gt, {0, 1, 2}, 1e-9);
  expectMatchesGroundTruth(res, R_gt, {3, 4, 5}, 1e-9);
  expectRotationNear(res.rotations[0], Mat3::Identity(), 1e-12);
  expectRotationNear(res.rotations[3], Mat3::Identity(), 1e-12);
}

// A node with no edges cannot be estimated. It defaults to the identity, or to a caller-supplied
// previous estimate -- which is what keeps a GNC iteration from snapping a node back to identity
// the moment all of its correspondences are rejected.
TEST(RotationSyncTest, UnconstrainedNodeFallsBackToInitialGuess) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<teaser::RotationSyncEdge> edges = {makeEdge(0, 1, R_gt, 20)};

  const auto plain = teaser::synchronizeRotations(3, edges);
  EXPECT_FALSE(plain.valid[2]);
  EXPECT_EQ(plain.component[2], -1);
  expectRotationNear(plain.rotations[2], Mat3::Identity(), 1e-12);

  const Mat3 previous = randomRotation();
  std::vector<Mat3> initial = {Mat3::Identity(), Mat3::Identity(), previous};
  const auto warm = teaser::synchronizeRotations(3, edges, teaser::RotationSyncParams(), &initial);

  EXPECT_FALSE(warm.valid[2]);
  expectRotationNear(warm.rotations[2], previous, 1e-12);
  // The estimated nodes are unaffected by the fallback.
  expectMatchesGroundTruth(warm, R_gt, {0, 1}, 1e-9);
}

TEST(RotationSyncTest, HandlesSingleEdgeAndDegenerateInputs) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation()};

  // Smallest real problem: two nodes, one edge. dim == 6, which is below the sparse solver's
  // nev < dim requirement, so this exercises the dense path.
  std::vector<teaser::RotationSyncEdge> one = {makeEdge(0, 1, R_gt, 15)};
  const auto res = teaser::synchronizeRotations(2, one);
  EXPECT_EQ(res.num_components, 1);
  expectMatchesGroundTruth(res, R_gt, {0, 1}, 1e-9);

  // No edges at all.
  const auto empty = teaser::synchronizeRotations(3, {});
  EXPECT_EQ(empty.num_components, 0);
  EXPECT_EQ(empty.rotations.size(), 3u);
  for (int p = 0; p < 3; ++p) {
    EXPECT_FALSE(empty.valid[p]);
    expectRotationNear(empty.rotations[p], Mat3::Identity(), 1e-12);
  }

  // No nodes at all.
  const auto none = teaser::synchronizeRotations(0, {});
  EXPECT_EQ(none.num_components, 0);
  EXPECT_TRUE(none.rotations.empty());

  // Malformed edges are skipped rather than crashing: out of range, self loop, and mismatched
  // column counts.
  teaser::RotationSyncEdge bad_dims = makeEdge(0, 1, R_gt, 10);
  bad_dims.w = Wts::Ones(1, 4);
  std::vector<teaser::RotationSyncEdge> bad = {makeEdge(0, 5, R_gt, 10), makeEdge(1, 1, R_gt, 10),
                                               bad_dims};
  bad[1].q = 1;
  bad[1].p = 1;
  const auto skipped = teaser::synchronizeRotations(2, bad);
  EXPECT_EQ(skipped.num_components, 0);
}

// The sparse and dense eigensolver paths must agree; only dense_max_dim differs between the runs.
TEST(RotationSyncTest, SparseAndDensePathsAgree) {
  constexpr int kNodes = 30; // dim == 90, above the default dense_max_dim of 60
  std::vector<Mat3> R_gt;
  for (int i = 0; i < kNodes; ++i) {
    R_gt.push_back(randomRotation());
  }
  std::vector<teaser::RotationSyncEdge> edges;
  for (int i = 0; i + 1 < kNodes; ++i) {
    edges.push_back(makeEdge(i, i + 1, R_gt, 12, 0.01));
  }
  for (int i = 0; i + 5 < kNodes; ++i) {
    edges.push_back(makeEdge(i, i + 5, R_gt, 12, 0.01)); // chords, for a healthy spectral gap
  }
  edges.push_back(makeEdge(0, kNodes - 1, R_gt, 12, 0.01));

  teaser::RotationSyncParams force_dense;
  force_dense.dense_max_dim = 100000;
  teaser::RotationSyncParams force_sparse;
  force_sparse.dense_max_dim = 0;

  const auto dense = teaser::synchronizeRotations(kNodes, edges, force_dense);
  const auto sparse = teaser::synchronizeRotations(kNodes, edges, force_sparse);

  ASSERT_EQ(dense.num_components, 1);
  ASSERT_EQ(sparse.num_components, 1);
  for (int p = 0; p < kNodes; ++p) {
    ASSERT_TRUE(dense.valid[p]);
    ASSERT_TRUE(sparse.valid[p]);
    expectProperRotation(sparse.rotations[p]);
    expectRotationNear(sparse.rotations[p], dense.rotations[p], 1e-8);
  }
  EXPECT_NEAR(sparse.spectral_gap[0], dense.spectral_gap[0], 1e-8);
  // Both should be close to ground truth despite the noise.
  std::vector<int> all(kNodes);
  for (int i = 0; i < kNodes; ++i) {
    all[i] = i;
  }
  expectMatchesGroundTruth(dense, R_gt, all, 5e-2);
}

TEST(RotationSyncTest, TimResidualNoiseBoundComposition) {
  // A TIM is a difference of two points, so each scan contributes 2 * delta, and both sides are
  // TIMs from real scans.
  EXPECT_DOUBLE_EQ(teaser::timResidualNoiseBound(0.01, 0.01), 0.04);
  EXPECT_DOUBLE_EQ(teaser::timResidualNoiseBound(0.01, 0.05), 0.12);
  // Symmetric in its arguments, and a noiseless scan still leaves the other scan's contribution.
  EXPECT_DOUBLE_EQ(teaser::timResidualNoiseBound(0.03, 0.07),
                   teaser::timResidualNoiseBound(0.07, 0.03));
  EXPECT_DOUBLE_EQ(teaser::timResidualNoiseBound(0.0, 0.02), 0.04);
}

// Per-scan bounds enter as 1/sigma_pq^2 on the edge weight. Uniform bounds put the SAME factor on
// every edge, and a common factor cancels out of D^{-1/2} B D^{-1/2}, so the answer must not move
// -- however large or small that common bound is.
TEST(RotationSyncTest, UniformNoiseBoundsAreANoOp) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(), randomRotation()};
  std::vector<teaser::RotationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, 25, 0.02), makeEdge(1, 2, R_gt, 25, 0.02),
      makeEdge(2, 3, R_gt, 25, 0.02), makeEdge(0, 3, R_gt, 25, 0.02)};

  const auto base = teaser::synchronizeRotations(4, edges);

  for (double delta : {1e-4, 0.05, 12.0}) {
    const std::vector<double> bounds(4, delta);
    const auto res =
        teaser::synchronizeRotations(4, edges, teaser::RotationSyncParams(), nullptr, &bounds);
    ASSERT_EQ(res.num_components, 1);
    for (int p = 0; p < 4; ++p) {
      expectRotationNear(res.rotations[p], base.rotations[p], 1e-9);
    }
  }
}

// Supplying per-scan bounds must be exactly equivalent to folding 1/sigma_pq^2 into the per-edge
// confidence by hand. This pins both the composition rule and where it is applied.
TEST(RotationSyncTest, NoiseBoundsMatchEquivalentConfidence) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(), randomRotation()};
  const std::vector<double> bounds = {0.01, 0.04, 0.20, 0.02}; // deliberately heterogeneous
  std::vector<teaser::RotationSyncEdge> edges = {
      makeEdge(0, 1, R_gt, 25, 0.02), makeEdge(1, 2, R_gt, 25, 0.02),
      makeEdge(2, 3, R_gt, 25, 0.02), makeEdge(0, 3, R_gt, 25, 0.02)};

  const auto via_bounds =
      teaser::synchronizeRotations(4, edges, teaser::RotationSyncParams(), nullptr, &bounds);

  std::vector<teaser::RotationSyncEdge> manual = edges;
  for (auto& e : manual) {
    const double sigma = teaser::timResidualNoiseBound(bounds[e.p], bounds[e.q]);
    e.confidence = 1.0 / (sigma * sigma);
  }
  const auto via_confidence = teaser::synchronizeRotations(4, manual);

  ASSERT_EQ(via_bounds.num_components, 1);
  for (int p = 0; p < 4; ++p) {
    expectRotationNear(via_bounds.rotations[p], via_confidence.rotations[p], 1e-12);
  }

  // And it genuinely changed the answer relative to ignoring the bounds.
  const auto unweighted = teaser::synchronizeRotations(4, edges);
  double max_diff = 0;
  for (int p = 0; p < 4; ++p) {
    max_diff = std::max(max_diff, (via_bounds.rotations[p] - unweighted.rotations[p]).norm());
  }
  EXPECT_GT(max_diff, 1e-6);
}

// The point of per-scan bounds: a scan known to be noisy should pull less on its neighbors, so the
// well-observed nodes come out closer to ground truth.
TEST(RotationSyncTest, NoisyScanIsDownweighted) {
  constexpr int kTrials = 20;
  constexpr double kCleanNoise = 0.005;
  constexpr double kDirtyNoise = 0.20;
  // 4-cycle 0-1-2-3-0 where node 2's observations are far noisier than everyone else's.
  const std::vector<double> bounds = {kCleanNoise, kCleanNoise, kDirtyNoise, kCleanNoise};

  double weighted_err = 0;
  double unweighted_err = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation(),
                              randomRotation()};
    std::vector<teaser::RotationSyncEdge> edges = {
        makeEdge(0, 1, R_gt, 25, kCleanNoise), makeEdge(1, 2, R_gt, 25, kDirtyNoise),
        makeEdge(2, 3, R_gt, 25, kDirtyNoise), makeEdge(0, 3, R_gt, 25, kCleanNoise)};

    const auto plain = teaser::synchronizeRotations(4, edges);
    const auto weighted =
        teaser::synchronizeRotations(4, edges, teaser::RotationSyncParams(), nullptr, &bounds);

    // Error on the three well-observed nodes, in the solver's gauge (node 0 == identity).
    const Mat3 anchor_inv = R_gt[0].transpose();
    for (int p : {1, 3}) {
      unweighted_err += (plain.rotations[p] - anchor_inv * R_gt[p]).norm();
      weighted_err += (weighted.rotations[p] - anchor_inv * R_gt[p]).norm();
    }
  }

  EXPECT_LT(weighted_err, unweighted_err)
      << "weighted=" << weighted_err << " unweighted=" << unweighted_err;
}

TEST(RotationSyncTest, NoiseBoundsDegenerateInputs) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<teaser::RotationSyncEdge> edges = {makeEdge(0, 1, R_gt, 20), makeEdge(1, 2, R_gt, 20),
                                                 makeEdge(0, 2, R_gt, 20)};

  // Both endpoints declared exactly noiseless -> sigma is 0 and the weight would be infinite, so
  // that edge is dropped. Here nodes 0 and 1 are both noiseless, killing edge (0,1); the remaining
  // two edges still connect everything.
  const std::vector<double> zeroed = {0.0, 0.0, 0.03};
  const auto dropped =
      teaser::synchronizeRotations(3, edges, teaser::RotationSyncParams(), nullptr, &zeroed);
  EXPECT_EQ(dropped.num_components, 1);
  expectMatchesGroundTruth(dropped, R_gt, {0, 1, 2}, 1e-9);

  // A wrongly sized vector is ignored rather than trusted or crashed on.
  const std::vector<double> wrong_size = {0.01, 0.01};
  const auto ignored =
      teaser::synchronizeRotations(3, edges, teaser::RotationSyncParams(), nullptr, &wrong_size);
  const auto base = teaser::synchronizeRotations(3, edges);
  ASSERT_EQ(ignored.num_components, 1);
  for (int p = 0; p < 3; ++p) {
    expectRotationNear(ignored.rotations[p], base.rotations[p], 1e-12);
  }
}

// A whole-edge confidence of zero removes the edge; a uniform positive confidence changes nothing,
// since B and the degrees scale together.
TEST(RotationSyncTest, EdgeConfidenceScalesAndDisables) {
  std::vector<Mat3> R_gt = {randomRotation(), randomRotation(), randomRotation()};
  std::vector<teaser::RotationSyncEdge> edges = {makeEdge(0, 1, R_gt, 20), makeEdge(1, 2, R_gt, 20),
                                                 makeEdge(0, 2, R_gt, 20)};

  const auto base = teaser::synchronizeRotations(3, edges);

  std::vector<teaser::RotationSyncEdge> scaled = edges;
  for (auto& e : scaled) {
    e.confidence = 7.5;
  }
  const auto scaled_res = teaser::synchronizeRotations(3, scaled);
  for (int p = 0; p < 3; ++p) {
    expectRotationNear(scaled_res.rotations[p], base.rotations[p], 1e-9);
  }

  std::vector<teaser::RotationSyncEdge> disabled = edges;
  disabled[1].confidence = 0.0; // drops edge (1,2); 0-1 and 0-2 still connect everything
  const auto disabled_res = teaser::synchronizeRotations(3, disabled);
  EXPECT_EQ(disabled_res.num_components, 1);
  expectMatchesGroundTruth(disabled_res, R_gt, {0, 1, 2}, 1e-9);
}
