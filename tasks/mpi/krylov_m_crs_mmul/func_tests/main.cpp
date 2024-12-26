#include <gtest/gtest.h>

#include <random>
#include <tuple>
#include <utility>

#include "../include/smmul_mpi.hpp"

using TestElementType = double;

class krylov_m_crs_mmul_test : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>, float>> {
 protected:
  boost::mpi::communicator world;

  static krylov_m_crs_mmul::Matrix<TestElementType> generate_random_matrix(size_t rows, size_t cols,
                                                                           TestElementType emin, TestElementType emax,
                                                                           float density) {
    const auto threshold = emin + ((emax - emin) * density);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(emin, emax);

    auto matrix = krylov_m_crs_mmul::Matrix<TestElementType>::create(rows, cols);
    std::generate(matrix.storage.begin(), matrix.storage.end(), [&]() {
      auto val = distr(gen);
      return (val < threshold) ? val : 0;
    });

    return matrix;
  }

  void peform_mul_test(size_t lrows, size_t lcols, size_t rcols, TestElementType emin, TestElementType emax,
                       float density) {
    std::pair<krylov_m_crs_mmul::Matrix<TestElementType>, krylov_m_crs_mmul::Matrix<TestElementType>> in;
    //
    std::pair<krylov_m_crs_mmul::CRSMatrix<TestElementType>, krylov_m_crs_mmul::CRSMatrix<TestElementType>> sin;
    krylov_m_crs_mmul::CRSMatrix<TestElementType> sout;

    auto taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      in = {generate_random_matrix(lrows, lcols, emin, emax, density),
            generate_random_matrix(lcols, rcols, emin, emax, density)};
      sin = {krylov_m_crs_mmul::CRSMatrix<TestElementType>(in.first),
             krylov_m_crs_mmul::CRSMatrix<TestElementType>(in.second)};

      //
      krylov_m_crs_mmul::fill_task_data(*taskData, sin.first, sin.second, sout);
    }

    //
    krylov_m_crs_mmul::TaskParallel<TestElementType> task(taskData);
    ASSERT_TRUE(task.validation());
    task.pre_processing();
    task.run();
    task.post_processing();

    if (world.rank() == 0) {
      EXPECT_EQ(sout.densify(), in.first * in.second);
    }
  }

  void perform_dimen_randomized_mul_test(size_t dmin, size_t dmax, TestElementType emin, TestElementType emax,
                                         float density) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(dmin, dmax);

    //
    peform_mul_test(distr(gen), distr(gen), distr(gen), emin, emax, density);
  }
};

TEST_P(krylov_m_crs_mmul_test, random_dimen_random_density) {
  const auto &[dimens, density] = GetParam();
  perform_dimen_randomized_mul_test(dimens.first, dimens.second, -128, 128, density);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_crs_mmul_test, krylov_m_crs_mmul_test,
    ::testing::Combine(
      testing::Values(
        std::make_pair(1, 2),
        std::make_pair(2, 4),
        std::make_pair(4, 8),
        std::make_pair(8, 16),
        std::make_pair(16, 24),
        std::make_pair(24, 32),
        std::make_pair(32, 48),
        std::make_pair(48, 64)
      ),
      testing::Values(0.00f, 0.05f, 0.10f, 0.15f, 0.25f, 0.30f, 0.50f, 0.64f, 0.75f, 0.80f, 0.90, 1.00f)
    )
);
// clang-format on

TEST_F(krylov_m_crs_mmul_test, random_3_7_13) { peform_mul_test(3, 7, 13, -128, 128, 0.5f); }

TEST_F(krylov_m_crs_mmul_test, bad_task_fail_validation) {
  krylov_m_crs_mmul::CRSMatrix<TestElementType> sout;

  krylov_m_crs_mmul::CRSMatrix<TestElementType> slhs{1, 3};
  krylov_m_crs_mmul::CRSMatrix<TestElementType> srhs{2, 2};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    krylov_m_crs_mmul::fill_task_data(*taskData, slhs, srhs, sout);
  }

  krylov_m_crs_mmul::TaskParallel<TestElementType> task(taskData);
  if (world.rank() == 0) {
    EXPECT_FALSE(task.validation());
  }
}

// clang-format off
using MulTestParam = std::tuple<
  krylov_m_crs_mmul::Matrix<TestElementType> /* lhs */,
  krylov_m_crs_mmul::Matrix<TestElementType> /* lhs */,
  krylov_m_crs_mmul::Matrix<TestElementType> /* ref */
>;
// clang-format on

class determined : public ::testing::TestWithParam<MulTestParam> {
 protected:
  boost::mpi::communicator world;

  void SetUp() override {
    const auto &[lhs, rhs, ref] = GetParam();

    ASSERT_TRUE(lhs.check_integrity());
    ASSERT_TRUE(rhs.check_integrity());
    ASSERT_TRUE(ref.check_integrity());
  }
};

TEST_P(determined, yields_correct_result) {
  const auto &[lhs, rhs, ref] = GetParam();

  const krylov_m_crs_mmul::CRSMatrix<TestElementType> slhs(lhs);
  const krylov_m_crs_mmul::CRSMatrix<TestElementType> srhs(rhs);
  krylov_m_crs_mmul::CRSMatrix<TestElementType> sout;

  //
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    krylov_m_crs_mmul::fill_task_data(*taskData, slhs, srhs, sout);
  }

  //
  krylov_m_crs_mmul::TaskParallel<TestElementType> task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    EXPECT_EQ(sout.densify(), ref);
  }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_crs_mmul_test, determined,
    ::testing::Values(
      MulTestParam(
        { 2, 2,
          {
            0, 0,
            0, 0
          }
        },
        { 2, 2,
          {
            0, 0,
            0, 0
          }
        },
        { 2, 2,
          {
            0, 0,
            0, 0
          }
        }
      ),
      //
      MulTestParam(
        { 2, 2,
          {
            4, 3,
            7, 5
          }
        },
        { 2, 2,
          {
            -28, 93,
            38, -126
          }
        },
        { 2, 2,
          {
            2, -6,
            -6, 21
          }
        }
      ),
      //
      MulTestParam(
        { 3, 3,
          {
            5, 3, -7,
            -1, 6, -3,
            2, -4, 1
          }
        },
        { 3, 3,
          {
            4, -1, 3,
            4, -2, -6,
            2, 0, 3
          }
        },
        { 3, 3,
          {
            18, -11, -24,
            14, -11, -48,
            -6, 6, 33
          }
        }
      ),
      //
      MulTestParam(
        { 2, 3,
          {
            2, 4, 1,
            1, 0, -2
          }
        },
        { 3, 3,
          {
            7, 3, 2,
            4, 1, 0,
            2, -1, 6
          }
        },
        { 2, 3,
          {
            32, 9, 10,
            3, 5, -10
          }
        }
      ),
      //
      MulTestParam(
        { 3, 3,
          {
            1,  0,  0,
            1, -1,  0,
            1,  0,  1
          }
        },
        { 3, 3,
          {
            1,  0,  0,
            1, -1,  0,
            -1,  0,  1
          }
        },
        { 3, 3,
          {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
          }
        }
      )
    )
);
// clang-format on
