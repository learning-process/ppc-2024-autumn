#include <gtest/gtest.h>

#include <concepts>
#include <random>
#include <tuple>

#include "../include/mmul_mpi.hpp"

using TestElementType = int64_t;

// clang-format off
using MatTestParam = std::tuple<
  size_t          /* lrows */,
  size_t          /* lcols */,
  size_t          /* rcols */,
  TestElementType /* emin */,
  TestElementType /* emax */
>;
// clang-format on

class krylov_m_matmul_strip_ha_vb_mpi_test : public ::testing::TestWithParam<MatTestParam> {
 protected:
  boost::mpi::communicator world;

  template <std::integral T>
  static krylov_m_matmul_strip_ha_vb_mpi::TMatrix<T> generate_random_matrix(size_t rows, size_t cols, T min, T max) {
    auto matrix = krylov_m_matmul_strip_ha_vb_mpi::TMatrix<T>::create(rows, cols);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(min, max);

    std::generate(matrix.storage.begin(), matrix.storage.end(), [&distr, &gen]() { return distr(gen); });

    return matrix;
  }

  template <typename T>
  void peform_random_test(size_t lrows, size_t lcols, size_t rcols, TestElementType emin, TestElementType emax) {
    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<T> lhs;
    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<T> rhs;
    //
    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<T> out;

    auto taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      lhs = generate_random_matrix<T>(lrows, lcols, emin, emax);
      ASSERT_TRUE(lhs.check_integrity());
      //
      rhs = generate_random_matrix<T>(lcols, rcols, emin, emax);
      ASSERT_TRUE(rhs.check_integrity());

      //
      krylov_m_matmul_strip_ha_vb_mpi::fill_task_data(*taskData, lhs, rhs, out);
    }

    //
    krylov_m_matmul_strip_ha_vb_mpi::TaskParallel<T> task(taskData);
    ASSERT_TRUE(task.validation());
    task.pre_processing();
    task.run();
    task.post_processing();

    if (world.rank() == 0) {
      decltype(out) ref_out;

      //
      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      krylov_m_matmul_strip_ha_vb_mpi::fill_task_data(*taskDataSeq, lhs, rhs, ref_out);

      //
      krylov_m_matmul_strip_ha_vb_mpi::TaskSequential<T> taskSeq(taskDataSeq);
      ASSERT_TRUE(taskSeq.validation());
      taskSeq.pre_processing();
      taskSeq.run();
      taskSeq.post_processing();

      EXPECT_EQ(out, ref_out);
    }
  }

  template <typename T>
  void peform_dimen_randomized_test(size_t dmin, size_t dmax, T emin, T emax) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(dmin, dmax);

    peform_random_test<T>(distr(gen), distr(gen), distr(gen), emin, emax);
  }
};

TEST_P(krylov_m_matmul_strip_ha_vb_mpi_test, yields_correct_result) {
  const auto &[lrows, lcols, rcols, emin, emax] = GetParam();
  peform_random_test<TestElementType>(lrows, lcols, rcols, emin, emax);
}

TEST_F(krylov_m_matmul_strip_ha_vb_mpi_test, yields_correct_result_random_dimen_16) {
  peform_dimen_randomized_test<TestElementType>(1, 16, -512, 512);
}
TEST_F(krylov_m_matmul_strip_ha_vb_mpi_test, yields_correct_result_random_dimen_32) {
  peform_dimen_randomized_test<TestElementType>(1, 32, -512, 512);
}
TEST_F(krylov_m_matmul_strip_ha_vb_mpi_test, yields_correct_result_random_dimen_64) {
  peform_dimen_randomized_test<TestElementType>(1, 64, -512, 512);
}
TEST_F(krylov_m_matmul_strip_ha_vb_mpi_test, yields_correct_result_random_dimen_128) {
  peform_dimen_randomized_test<TestElementType>(1, 64, -512, 512);
}

TEST_F(krylov_m_matmul_strip_ha_vb_mpi_test, bad_task_fail_validation) {
  if (world.rank() == 0) {
    krylov_m_matmul_strip_ha_vb_mpi::TaskSequential<TestElementType>::Matrix out;

    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<TestElementType> lhs{2, 5};
    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<TestElementType> rhs{3, 2};

    auto taskData = std::make_shared<ppc::core::TaskData>();
    krylov_m_matmul_strip_ha_vb_mpi::fill_task_data(*taskData, lhs, rhs, out);

    krylov_m_matmul_strip_ha_vb_mpi::TaskSequential<TestElementType> task(taskData);
    EXPECT_FALSE(task.validation());
  }
}

TEST_F(krylov_m_matmul_strip_ha_vb_mpi_test, bad_channels_fail_validation) {
  if (world.rank() == 0) {
    krylov_m_matmul_strip_ha_vb_mpi::TaskSequential<TestElementType>::Matrix out;

    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<TestElementType> lhs{2, 3};
    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<TestElementType> rhs{3, 2};
    uint8_t trash{};

    // fill task data...
    auto taskData = std::make_shared<ppc::core::TaskData>();
    krylov_m_matmul_strip_ha_vb_mpi::fill_task_data(*taskData, lhs, rhs, out);
    // ...and damage it
    taskData->outputs.emplace_back(&trash);

    krylov_m_matmul_strip_ha_vb_mpi::TaskSequential<TestElementType> task(taskData);
    EXPECT_FALSE(task.validation());
  }
}

TEST_F(krylov_m_matmul_strip_ha_vb_mpi_test, bad_prealloc_fail_validation) {
  if (world.rank() == 0) {
    krylov_m_matmul_strip_ha_vb_mpi::TaskSequential<TestElementType>::Matrix out;

    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<TestElementType> lhs{2, 3};
    krylov_m_matmul_strip_ha_vb_mpi::TMatrix<TestElementType> rhs{3, 2};

    // fill task data...
    auto taskData = std::make_shared<ppc::core::TaskData>();
    krylov_m_matmul_strip_ha_vb_mpi::fill_task_data(*taskData, lhs, rhs, out);
    // ...and damage it
    taskData->outputs_count[0] += 2;

    krylov_m_matmul_strip_ha_vb_mpi::TaskSequential<TestElementType> task(taskData);
    EXPECT_FALSE(task.validation());
  }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_matmul_strip_ha_vb_mpi_test_fixed_sizes, krylov_m_matmul_strip_ha_vb_mpi_test,
    ::testing::Values(
      MatTestParam(1, 1, 1, -512, 512),
      MatTestParam(1, 2, 1, -512, 512),
      MatTestParam(1, 3, 1, -512, 512),
      //
      MatTestParam(2, 2, 2, -512, 512),
      MatTestParam(2, 2, 4, -512, 512),
      MatTestParam(2, 4, 4, -512, 512),
      MatTestParam(4, 4, 4, -512, 512),
      MatTestParam(2, 2, 5, -512, 512),
      MatTestParam(2, 5, 5, -512, 512),
      MatTestParam(5, 5, 5, -512, 512),
      //
      MatTestParam(3, 3, 3, -512, 512),
      MatTestParam(3, 3, 7, -512, 512),
      MatTestParam(3, 7, 7, -512, 512),
      MatTestParam(7, 7, 7, -512, 512),
      //
      MatTestParam(3, 3, 1, -512, 512),
      MatTestParam(3, 1, 1, -512, 512),
      //
      MatTestParam(5, 5, 5, -512, 512),
      MatTestParam(5, 5, 9, -512, 512),
      MatTestParam(5, 9, 9, -512, 512),
      MatTestParam(9, 9, 9, -512, 512),
      //
      MatTestParam(8, 8, 8, -512, 512),
      MatTestParam(8, 8, 16, -512, 512),
      MatTestParam(8, 16, 16, -512, 512),
      MatTestParam(16, 16, 16, -512, 512),
      MatTestParam(128, 128, 128, -512, 512),
      //
      MatTestParam(5, 3, 4, -512, 512),
      MatTestParam(8, 11, 16, -512, 512),
      MatTestParam(15, 12, 16, -512, 512)
    )
);
// clang-format on
