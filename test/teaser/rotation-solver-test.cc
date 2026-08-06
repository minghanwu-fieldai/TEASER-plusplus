/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "teaser/registration.h"
#include "teaser/utils.h"
#include "test_utils.h"

TEST(RotationSolverTest, FGRRotation) {
  double ALLOWED_ROTATION_ERROR = 1e-5;
  // Problem 1: Identity
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points = src_points;

    // Set up FGR
    teaser::FastGlobalRegistrationSolver::Params params{1000, 0.0337, 1.4, 1e-3};
    teaser::FastGlobalRegistrationSolver fgr_solver(params);

    Eigen::Matrix3d result;
    fgr_solver.solveForRotation(src_points, dst_points, &result, nullptr);
    Eigen::Matrix3d ref_result;
    ref_result.setIdentity();
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_result << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE((result - ref_result).norm() < ALLOWED_ROTATION_ERROR);
  }
  // Problem 2: Random rotation around x / y / z axis
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix3d ref_R;
    std::uniform_real_distribution<double> unif(0, 2 * M_PI);
    std::default_random_engine re;

    // Prepare solver
    teaser::FastGlobalRegistrationSolver::Params params{1000, 0.0337, 1.4, 1e-3};
    teaser::FastGlobalRegistrationSolver fgr_solver(params);
    Eigen::Matrix3d R;
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points;

    // Rotation around x
    double theta = unif(re);
    // clang-format off
    ref_R << 1, 0,               0,
             0, std::cos(theta), -std::sin(theta),
             0, std::sin(theta), std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    fgr_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around y
    // clang-format off
    ref_R << std::cos(theta), 0, std::sin(theta),
             0,               1, 0,
             -std::sin(theta),0, std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    fgr_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around z
    ref_R << std::cos(theta), -std::sin(theta), 0, std::sin(theta), std::cos(theta), 0, 0, 0, 1;
    // clang-format on
    dst_points = ref_R * src_points;
    fgr_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);
  }
  // Problem 3: A more complex case
  {
    // Read in data
    std::ifstream source_file("./data/registration_test/rotation_only_src.csv");
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_points =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, 3>(source_file);
    Eigen::Matrix<double, 3, Eigen::Dynamic> src = source_points.transpose();

    // Perform Arbitrary rotation
    Eigen::Matrix3d expected_R;
    // clang-format off
    expected_R << 0.997379773225804, -0.019905935977315, -0.069551000516966,
                  0.013777311189888, 0.996068297974922, -0.087510750572249,
                  0.071019530105605, 0.086323226782879, 0.993732623426126;
    // clang-format on
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = expected_R * src;

    // Set up FGR
    // Since we have no noise, 1 iteration should give us the optimal solution.
    teaser::FastGlobalRegistrationSolver::Params params{1, 0.025, 1.4, 1e-3};
    teaser::FastGlobalRegistrationSolver fgr_solver(params);

    Eigen::Matrix3d result;
    fgr_solver.solveForRotation(src, dst, &result, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << expected_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE(teaser::test::getAngularError(expected_R, result) < ALLOWED_ROTATION_ERROR);
  }
}

TEST(RotationSolverTest, GNCTLS) {
  double ALLOWED_ROTATION_ERROR = 1e-5;
  // Problem 1: Identity
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points = src_points;

    // Set up GNC-TLS solver
    teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-3};
    teaser::GNCTLSRotationSolver tls_solver(params);

    Eigen::Matrix3d result;
    tls_solver.solveForRotation(src_points, dst_points, &result, nullptr);
    Eigen::Matrix3d ref_result;
    ref_result.setIdentity();
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_result << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE((result - ref_result).norm() < ALLOWED_ROTATION_ERROR);
  }
  // Problem 2: Random rotation around x / y / z axis
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix3d ref_R;
    std::uniform_real_distribution<double> unif(0, 2 * M_PI);
    std::default_random_engine re;

    // Set up GNC-TLS solver
    teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-3};
    teaser::GNCTLSRotationSolver tls_solver(params);
    Eigen::Matrix3d R;
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points;

    // Rotation around x
    double theta = unif(re);
    // clang-format off
    ref_R << 1, 0,               0,
        0, std::cos(theta), -std::sin(theta),
        0, std::sin(theta), std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    tls_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around y
    // clang-format off
    ref_R << std::cos(theta), 0, std::sin(theta),
        0,                    1, 0,
        -std::sin(theta),     0, std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    tls_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around z
    // clang-format off
    ref_R << std::cos(theta), -std::sin(theta), 0,
             std::sin(theta), std::cos(theta),  0,
             0,               0,                1;
    // clang-format on
    dst_points = ref_R * src_points;
    tls_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);
  }
  // Problem 3: A more complex case
  {
    // Read in data
    std::ifstream source_file("./data/registration_test/rotation_only_src.csv");
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_points =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, 3>(source_file);
    Eigen::Matrix<double, 3, Eigen::Dynamic> src = source_points.transpose();

    // Perform Arbitrary rotation
    Eigen::Matrix3d expected_R;
    // clang-format off
    expected_R << 0.997379773225804, -0.019905935977315, -0.069551000516966,
                  0.013777311189888, 0.996068297974922, -0.087510750572249,
                  0.071019530105605, 0.086323226782879, 0.993732623426126;
    // clang-format on
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = expected_R * src;

    // Set up TLS
    teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-3};
    teaser::GNCTLSRotationSolver tls_solver(params);

    Eigen::Matrix3d result;
    tls_solver.solveForRotation(src, dst, &result, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << expected_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE(teaser::test::getAngularError(expected_R, result) < ALLOWED_ROTATION_ERROR);
  }
}

// ============================================================================
// Soft tilt (pitch/roll) prior -- GNCRotationSolver::Params::tilt_prior_eta
// ============================================================================

namespace {

/**
 * Combined pitch/roll deviation of a rotation, i.e. the angle between R*z_hat and z_hat.
 * This is exactly the quantity the tilt prior penalizes: R(2,2) = cos(pitch)*cos(roll).
 */
double tiltAngle(const Eigen::Matrix3d& R) {
  return std::acos(std::fmin(std::fmax(R(2, 2), -1.0), 1.0));
}

/** Yaw extracted from a rotation, i.e. the component the tilt prior must leave alone. */
double yawAngle(const Eigen::Matrix3d& R) { return std::atan2(R(1, 0), R(0, 0)); }

Eigen::Matrix3d makeYPR(double yaw, double pitch, double roll) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

/** Deterministic pseudo-random unit-ish points, so the sweeps below are reproducible. */
Eigen::Matrix<double, 3, Eigen::Dynamic> deterministicPoints(int n, unsigned seed = 4242) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  Eigen::Matrix<double, 3, Eigen::Dynamic> pts(3, n);
  for (int i = 0; i < n; ++i) {
    pts.col(i) << unif(gen), unif(gen), unif(gen);
  }
  return pts;
}

/**
 * Params for the tilt-prior tests.
 *
 * The noise bound is set deliberately large relative to the residuals so that the GNC-TLS mu
 * initialization rule (mu = 1/(2*max_residual/noise_bound^2 - 1)) yields mu <= 0 and the solver
 * terminates after its first R-step with all line-process weights still at 1. That isolates the
 * *penalized least-squares* behavior, which is what the prior itself is about, from the robust
 * outlier-rejection behavior (covered separately by TiltPriorSurvivesOutliers below).
 */
teaser::GNCTLSRotationSolver::Params tiltTestParams(double eta) {
  teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 5.0};
  params.tilt_prior_eta = eta;
  return params;
}

} // namespace

// The default (eta = 0) must reproduce the unpenalized solver exactly, including for ground truth
// with substantial pitch and roll.
TEST(RotationSolverTest, TiltPriorDisabledByDefault) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);
  const Eigen::Matrix3d ref_R = makeYPR(0.6, 0.35, -0.25);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-3};
  EXPECT_EQ(params.tilt_prior_eta, 0.0);
  EXPECT_TRUE(params.up_src.isApprox(Eigen::Vector3d::UnitZ()));
  EXPECT_TRUE(params.up_dst.isApprox(Eigen::Vector3d::UnitZ()));

  teaser::GNCTLSRotationSolver solver(params);
  Eigen::Matrix3d R;
  solver.solveForRotation(src, dst, &R, nullptr);
  EXPECT_LT(teaser::test::getAngularError(ref_R, R), 1e-8);
  EXPECT_NEAR(tiltAngle(R), tiltAngle(ref_R), 1e-8);
}

// A pure-yaw ground truth zeroes both the data term and the penalty simultaneously, so it stays the
// global optimum of the penalized problem for *every* eta. This is the exact statement of "the
// prior does not touch yaw".
TEST(RotationSolverTest, TiltPriorLeavesPureYawUntouched) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);
  const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.0, 0.0);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  for (double eta : {0.0, 0.1, 0.5, 10.0, 1e6}) {
    teaser::GNCTLSRotationSolver solver(tiltTestParams(eta));
    Eigen::Matrix3d R;
    solver.solveForRotation(src, dst, &R, nullptr);
    EXPECT_LT(teaser::test::getAngularError(ref_R, R), 1e-8) << "eta = " << eta;
    EXPECT_NEAR(yawAngle(R), 0.7, 1e-8) << "eta = " << eta;
  }
}

// Increasing eta must monotonically shrink the recovered tilt. For the exact minimizer of a
// penalized objective this is guaranteed, so only numerical slack is allowed.
TEST(RotationSolverTest, TiltPriorShrinksPitchRoll) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);
  const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.30, -0.20);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  const std::vector<double> etas = {0.0, 0.05, 0.1, 0.5, 2.0, 10.0};
  std::vector<double> tilts;
  for (double eta : etas) {
    teaser::GNCTLSRotationSolver solver(tiltTestParams(eta));
    Eigen::Matrix3d R;
    solver.solveForRotation(src, dst, &R, nullptr);
    tilts.push_back(tiltAngle(R));
    std::cout << "eta = " << eta << "  tilt = " << tilts.back()
              << " rad, yaw = " << yawAngle(R) << " rad" << std::endl;
  }

  // eta = 0 recovers the ground-truth tilt.
  EXPECT_NEAR(tilts.front(), tiltAngle(ref_R), 1e-8);
  // Monotonically non-increasing in eta.
  for (size_t i = 1; i < tilts.size(); ++i) {
    EXPECT_LE(tilts[i], tilts[i - 1] + 1e-9) << "eta = " << etas[i];
  }
  // And the effect is substantial, not just nominal.
  EXPECT_LT(tilts.back(), 0.25 * tilts.front());
}

// The eta -> infinity limit is the yaw-only solution, which for a z-axis prior is exactly the SO(2)
// fit to the XY components (the z residual does not depend on yaw). That is what svdRot2d computes,
// i.e. the same solution QUATRO produces as a hard constraint.
TEST(RotationSolverTest, TiltPriorLargeEtaApproachesYawOnly) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);
  const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.30, -0.20);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  teaser::GNCTLSRotationSolver solver(tiltTestParams(1e8));
  Eigen::Matrix3d R;
  solver.solveForRotation(src, dst, &R, nullptr);

  // Essentially zero pitch/roll.
  EXPECT_LT(tiltAngle(R), 1e-3);

  // Matches the yaw-only least-squares reference.
  Eigen::Matrix<double, 1, Eigen::Dynamic> ones =
      Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1, src.cols());
  Eigen::Matrix2d R2 = teaser::utils::svdRot2d(src.topRows(2), dst.topRows(2), ones);
  EXPECT_NEAR(yawAngle(R), std::atan2(R2(1, 0), R2(0, 0)), 1e-3);
}

// lambda = eta * N * L^2 is normalized by the measurement mass, so the estimate must not depend on
// the units of the input.
TEST(RotationSolverTest, TiltPriorScaleInvariance) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);
  const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.30, -0.20);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  Eigen::Matrix3d R_unit, R_scaled;
  teaser::GNCTLSRotationSolver solver_a(tiltTestParams(0.5));
  solver_a.solveForRotation(src, dst, &R_unit, nullptr);

  // Scale both clouds by a large constant; noise_bound is scaled with them so the GNC path is
  // unchanged too.
  const double c = 100.0;
  teaser::GNCTLSRotationSolver::Params scaled_params = tiltTestParams(0.5);
  scaled_params.noise_bound *= c;
  teaser::GNCTLSRotationSolver solver_b(scaled_params);
  solver_b.solveForRotation(c * src, c * dst, &R_scaled, nullptr);

  EXPECT_LT(teaser::test::getAngularError(R_unit, R_scaled), 1e-8);
}

// Asymmetric up vectors express a known *relative* tilt (e.g. per-frame IMU gravity). When the prior
// agrees with the data, even an enormous eta must not distort the answer.
TEST(RotationSolverTest, TiltPriorAsymmetricUpDoesNotDistort) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);
  const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.30, -0.20);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  // At eta this large the prior term dominates the correlation matrix by ~1e6, so the recoverable
  // accuracy is conditioning-limited at ~2e-8 rather than machine precision. Exact at moderate eta.
  teaser::GNCTLSRotationSolver::Params params = tiltTestParams(1e6);
  params.up_src = Eigen::Vector3d::UnitZ();
  params.up_dst = ref_R * Eigen::Vector3d::UnitZ(); // gravity as seen in the dst frame
  teaser::GNCTLSRotationSolver solver(params);

  Eigen::Matrix3d R;
  solver.solveForRotation(src, dst, &R, nullptr);
  EXPECT_LT(teaser::test::getAngularError(ref_R, R), 1e-6);

  // Up vectors need not be normalized by the caller.
  params.up_src = 7.0 * Eigen::Vector3d::UnitZ();
  params.up_dst = 3.0 * (ref_R * Eigen::Vector3d::UnitZ());
  teaser::GNCTLSRotationSolver solver_unnormalized(params);
  Eigen::Matrix3d R_unnormalized;
  solver_unnormalized.solveForRotation(src, dst, &R_unnormalized, nullptr);
  EXPECT_LT(teaser::test::getAngularError(ref_R, R_unnormalized), 1e-6);

  // Moderate eta is exact.
  params = tiltTestParams(10.0);
  params.up_src = Eigen::Vector3d::UnitZ();
  params.up_dst = ref_R * Eigen::Vector3d::UnitZ();
  teaser::GNCTLSRotationSolver solver_moderate(params);
  Eigen::Matrix3d R_moderate;
  solver_moderate.solveForRotation(src, dst, &R_moderate, nullptr);
  EXPECT_LT(teaser::test::getAngularError(ref_R, R_moderate), 1e-12);
}

// The prior must not leak into the robust machinery: outliers are still rejected, the inlier mask
// still has exactly one entry per input correspondence, and the tilt still shrinks.
TEST(RotationSolverTest, TiltPriorSurvivesOutliers) {
  const int N = 40;
  const int num_outliers = 10;
  Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(N);
  const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.30, -0.20);
  Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  // Corrupt the last few correspondences.
  Eigen::Matrix<double, 3, Eigen::Dynamic> junk = deterministicPoints(num_outliers, 99);
  dst.rightCols(num_outliers) = 5.0 * junk;

  double tilt_unpenalized = 0.0;
  for (double eta : {0.0, 0.5}) {
    // noise_bound must be wide enough to absorb the residuals the prior induces on true inliers,
    // otherwise GNC rejects them too -- see tilt_prior_eta's calibration note. The 5x-corrupted
    // outliers remain far outside this bound.
    teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 0.5};
    params.tilt_prior_eta = eta;
    teaser::GNCTLSRotationSolver solver(params);

    Eigen::Matrix3d R;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers(1, N);
    solver.solveForRotation(src, dst, &R, &inliers);

    // One mask entry per input correspondence -- no phantom entry from the prior.
    ASSERT_EQ(inliers.cols(), N);
    // All the injected outliers are rejected.
    for (int i = N - num_outliers; i < N; ++i) {
      EXPECT_FALSE(inliers(0, i)) << "outlier " << i << " not rejected, eta = " << eta;
    }
    // A healthy majority of the true inliers is kept.
    EXPECT_GT(inliers.leftCols(N - num_outliers).count(), (N - num_outliers) / 2);

    if (eta == 0.0) {
      tilt_unpenalized = tiltAngle(R);
      EXPECT_LT(teaser::test::getAngularError(ref_R, R), 1e-4);
    } else {
      EXPECT_LT(tiltAngle(R), tilt_unpenalized);
    }
  }
}

TEST(RotationSolverTest, TiltPriorNegativeEtaThrows) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(10);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = src;
  Eigen::Matrix3d R;

  // GNC-TLS solver
  {
    teaser::GNCTLSRotationSolver solver(tiltTestParams(-1e-9));
    EXPECT_THROW(solver.solveForRotation(src, dst, &R, nullptr), std::invalid_argument);
  }
  // FGR solver
  {
    teaser::FastGlobalRegistrationSolver::Params params{100, 1e-12, 1.4, 5.0};
    params.tilt_prior_eta = -1.0;
    teaser::FastGlobalRegistrationSolver solver(params);
    EXPECT_THROW(solver.solveForRotation(src, dst, &R, nullptr), std::invalid_argument);
  }
  // Through RobustRegistrationSolver, which validates at construction time.
  {
    teaser::RobustRegistrationSolver::Params params;
    params.rotation_tilt_prior_eta = -0.5;
    EXPECT_THROW(teaser::RobustRegistrationSolver solver(params), std::invalid_argument);
  }
  // A degenerate up vector is only an error when the prior is actually enabled.
  {
    teaser::GNCTLSRotationSolver::Params params = tiltTestParams(0.5);
    params.up_src = Eigen::Vector3d::Zero();
    teaser::GNCTLSRotationSolver solver(params);
    EXPECT_THROW(solver.solveForRotation(src, dst, &R, nullptr), std::invalid_argument);

    params.tilt_prior_eta = 0.0;
    teaser::GNCTLSRotationSolver disabled_solver(params);
    EXPECT_NO_THROW(disabled_solver.solveForRotation(src, dst, &R, nullptr));
  }
}

// FGR shares the same prior; its annealing differs, so assert the qualitative behavior plus the
// exact pure-yaw invariance.
TEST(RotationSolverTest, TiltPriorFGR) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);

  // Pure yaw: untouched for any eta.
  {
    const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.0, 0.0);
    const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;
    for (double eta : {0.0, 0.5, 1e6}) {
      teaser::FastGlobalRegistrationSolver::Params params{100, 1e-12, 1.4, 1e-2};
      params.tilt_prior_eta = eta;
      teaser::FastGlobalRegistrationSolver solver(params);
      Eigen::Matrix3d R;
      solver.solveForRotation(src, dst, &R, nullptr);
      EXPECT_LT(teaser::test::getAngularError(ref_R, R), 1e-6) << "eta = " << eta;
    }
  }

  // With pitch/roll: the prior shrinks the tilt, and a huge eta drives it to zero.
  {
    const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.30, -0.20);
    const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;
    std::vector<double> tilts;
    for (double eta : {0.0, 0.5, 1e8}) {
      teaser::FastGlobalRegistrationSolver::Params params{100, 1e-12, 1.4, 1e-2};
      params.tilt_prior_eta = eta;
      teaser::FastGlobalRegistrationSolver solver(params);
      Eigen::Matrix3d R;
      solver.solveForRotation(src, dst, &R, nullptr);
      tilts.push_back(tiltAngle(R));
      std::cout << "FGR eta = " << eta << "  tilt = " << tilts.back() << " rad" << std::endl;
    }
    EXPECT_NEAR(tilts[0], tiltAngle(ref_R), 1e-6);
    EXPECT_LT(tilts[1], tilts[0]);
    EXPECT_LT(tilts[2], 1e-3);
  }
}

// End-to-end through RobustRegistrationSolver, exercising the Params plumbing and the TIM path.
TEST(RotationSolverTest, TiltPriorThroughRobustRegistrationSolver) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(30);
  const Eigen::Matrix3d ref_R = makeYPR(0.5, 0.25, -0.15);
  const Eigen::Vector3d ref_t(1.0, -2.0, 0.5);
  Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;
  dst.colwise() += ref_t;

  auto make_params = [](double eta) {
    teaser::RobustRegistrationSolver::Params params;
    // Wide enough that the prior can bend the rotation without GNC rejecting the data; see the
    // calibration note on tilt_prior_eta.
    params.noise_bound = 0.1;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 1e-12;
    params.rotation_tilt_prior_eta = eta;
    return params;
  };

  // eta = 0 recovers the ground truth.
  teaser::RobustRegistrationSolver solver_off(make_params(0.0));
  auto sol_off = solver_off.solve(src, dst);
  ASSERT_TRUE(sol_off.valid);
  EXPECT_LT(teaser::test::getAngularError(ref_R, sol_off.rotation), 1e-6);
  EXPECT_LT((sol_off.translation - ref_t).norm(), 1e-6);
  EXPECT_NEAR(tiltAngle(sol_off.rotation), tiltAngle(ref_R), 1e-6);

  // eta > 0 pulls the tilt down while yaw and translation stay sane.
  teaser::RobustRegistrationSolver solver_on(make_params(0.2));
  auto sol_on = solver_on.solve(src, dst);
  ASSERT_TRUE(sol_on.valid);
  EXPECT_LT(tiltAngle(sol_on.rotation), 0.85 * tiltAngle(sol_off.rotation));
  EXPECT_NEAR(yawAngle(sol_on.rotation), yawAngle(ref_R), 0.1);
  EXPECT_TRUE(sol_on.rotation.allFinite());
  EXPECT_TRUE(sol_on.translation.allFinite());
  EXPECT_LT((sol_on.translation - ref_t).norm(), 0.1);
}

// Pins the documented failure mode: an eta too large for the noise bound makes every residual look
// like an outlier, GNC zeroes all the weights, and the rotation ends up determined by the prior
// alone (correct up axis, arbitrary yaw). The solver warns; this test documents that it happens so
// the behavior is not mistaken for a regression.
TEST(RotationSolverTest, TiltPriorTooStrongForNoiseBoundRejectsEverything) {
  const Eigen::Matrix<double, 3, Eigen::Dynamic> src = deterministicPoints(20);
  const Eigen::Matrix3d ref_R = makeYPR(0.7, 0.30, -0.20);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> dst = ref_R * src;

  // A tight noise bound cannot pay for the tilt correction eta = 0.5 demands.
  teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-2};
  params.tilt_prior_eta = 0.5;
  teaser::GNCTLSRotationSolver solver(params);

  Eigen::Matrix3d R;
  Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers(1, src.cols());
  solver.solveForRotation(src, dst, &R, &inliers);

  EXPECT_EQ(inliers.count(), 0);
  // The up axis is still honored...
  EXPECT_LT(tiltAngle(R), 1e-9);
  // ...but the answer is nowhere near the ground truth.
  EXPECT_GT(teaser::test::getAngularError(ref_R, R), 0.1);

  // The same eta with a noise bound wide enough to pay for the correction behaves properly.
  params.noise_bound = 0.5;
  teaser::GNCTLSRotationSolver ok_solver(params);
  Eigen::Matrix3d R_ok;
  Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers_ok(1, src.cols());
  ok_solver.solveForRotation(src, dst, &R_ok, &inliers_ok);
  EXPECT_EQ(inliers_ok.count(), src.cols());
  EXPECT_LT(tiltAngle(R_ok), tiltAngle(ref_R));
  EXPECT_GT(tiltAngle(R_ok), 0.0);
}
