#include "../include/matrix.hpp"

#include <gtest/gtest.h>

#include <random>
#include <tuple>

using TestElementType = int16_t;

//

// clang-format off
using DenseMulTestParam = std::tuple<
  krylov_m_crs_mmul::Matrix<TestElementType> /* lhs */,
  krylov_m_crs_mmul::Matrix<TestElementType> /* lhs */,
  krylov_m_crs_mmul::Matrix<TestElementType> /* ref */
>;
// clang-format on

class dense_mul : public ::testing::TestWithParam<DenseMulTestParam> {
 protected:
  void SetUp() override {
    const auto &[lhs, rhs, ref] = GetParam();

    ASSERT_TRUE(lhs.check_integrity());
    ASSERT_TRUE(rhs.check_integrity());
    ASSERT_TRUE(ref.check_integrity());
    //
    ASSERT_TRUE(ref.rows == lhs.rows);
    ASSERT_TRUE(ref.cols == rhs.rows);
  }
};

TEST_P(dense_mul, yields_correct_result) {
  const auto &[lhs, rhs, ref] = GetParam();
  EXPECT_EQ(lhs * rhs, ref);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_crs_mmul_test, dense_mul,
    ::testing::Values(
      DenseMulTestParam(
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
      DenseMulTestParam(
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
      DenseMulTestParam(
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
      )
    )
);
// clang-format on

//

// clang-format off
using SparseConversionTestParam = std::tuple<
  krylov_m_crs_mmul::Matrix<TestElementType>    /* dense  */,
  krylov_m_crs_mmul::CRSMatrix<TestElementType> /* sparse_ref */
>;
// clang-format on

//

class sparse_conversion : public ::testing::TestWithParam<SparseConversionTestParam> {
  void SetUp() override {
    const auto &[dense, sparse_ref] = GetParam();

    ASSERT_TRUE(dense.check_integrity());
    //
    ASSERT_TRUE(dense.rows == sparse_ref.rows());
    ASSERT_TRUE(dense.cols == sparse_ref.cols());
  }
};

TEST_P(sparse_conversion, conversion_is_correct) {
  const auto &[dense, sparse_ref] = GetParam();
  EXPECT_EQ(krylov_m_crs_mmul::CRSMatrix<TestElementType>(dense), sparse_ref);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_crs_mmul_test, sparse_conversion,
    ::testing::Values(
      SparseConversionTestParam(
        { 7, 5,
          {
            8, 0, 2, 0, 0,
            0, 0, 5, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 7, 1, 2,
            0, 0, 0, 0, 0,
            0, 0, 0, 9, 0
          }
        },
        {
            { 0, 2, 3, 3, 3, 6, 6, 7}, 
            { 0, 2, 2, 2, 3, 4, 3 },
            { 8, 2, 5, 7, 1, 2, 9 },
            5
        }
      )
    )
);
// clang-format on

//

// clang-format off
using SparseTranspositionTestParam = std::tuple<
  krylov_m_crs_mmul::Matrix<TestElementType> /* dense_mat */,
  krylov_m_crs_mmul::Matrix<TestElementType> /* dense_ref */
>;
// clang-format on

//

class sparse_transposition : public ::testing::TestWithParam<SparseTranspositionTestParam> {};

TEST_P(sparse_transposition, conversion_is_correct) {
  const auto &[dense_mat, dense_ref] = GetParam();
  EXPECT_EQ(krylov_m_crs_mmul::CRSMatrix<TestElementType>(dense_mat).transpose(),
            krylov_m_crs_mmul::CRSMatrix<TestElementType>(dense_ref));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_crs_mmul_test, sparse_transposition,
    ::testing::Values(
      SparseTranspositionTestParam(
        { 7, 5,
          {
            8, 0, 2, 0, 0,
            0, 0, 5, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 7, 1, 2,
            0, 0, 0, 0, 0,
            0, 0, 0, 9, 0
          }
        },
        { 5, 7,
          {
            8, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            2, 5, 0, 0, 7, 0, 0,
            0, 0, 0, 0, 1, 0, 9,
            0, 0, 0, 0, 2, 0, 0
          }
        }
      )
    )
);
// clang-format on

//

class krylov_m_crs_mmul_test_generic : public ::testing::TestWithParam<krylov_m_crs_mmul::Matrix<TestElementType>> {
 public:
  static krylov_m_crs_mmul::Matrix<TestElementType> generate_random_matrix(size_t rows, size_t cols,
                                                                           TestElementType emin, TestElementType emax,
                                                                           float density) {
    const TestElementType threshold = emin + ((emax - emin) * density);

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
  static krylov_m_crs_mmul::Matrix<TestElementType> generate_random_size_matrix(size_t dmin, size_t dmax,
                                                                                TestElementType emin,
                                                                                TestElementType emax, float density) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(dmin, dmax);

    return generate_random_matrix(distr(gen), distr(gen), emin, emax, density);
  }
};

TEST_P(krylov_m_crs_mmul_test_generic, random_sparse_tests_conversion_is_idempotent) {
  const auto dense = GetParam();
  const krylov_m_crs_mmul::CRSMatrix<TestElementType> sparse(dense);
  EXPECT_EQ(sparse.densify(), dense);
}

TEST_P(krylov_m_crs_mmul_test_generic, random_sparse_tests_transposition_is_idempotent) {
  const auto dense = GetParam();
  const krylov_m_crs_mmul::CRSMatrix<TestElementType> sparse(dense);
  const auto transposed = sparse.transpose();
  EXPECT_EQ(transposed.transpose(), sparse);
}

INSTANTIATE_TEST_SUITE_P(
    krylov_m_crs_mmul_test_generic, krylov_m_crs_mmul_test_generic,
    ::testing::Values(krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.00f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.10f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.25f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.30f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.50f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.64f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.75f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 0.90f),
                      krylov_m_crs_mmul_test_generic::generate_random_size_matrix(1, 16, -128, 128, 1.00f)));
