#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "../include/mmul_seq.hpp"

using TestElementType = int64_t;

// clang-format off
using MatTestParam = std::tuple<
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>::Matrix /* lhs */,
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>::Matrix /* lhs */,
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>::Matrix /* ref */
>;
// clang-format on

class krylov_m_matmul_strip_ha_vb_seq_test : public ::testing::TestWithParam<MatTestParam> {
 protected:
};

TEST_P(krylov_m_matmul_strip_ha_vb_seq_test, yields_correct_result) {
  const auto &[lhs, rhs, ref] = GetParam();

  ASSERT_TRUE(lhs.check_integrity());
  ASSERT_TRUE(rhs.check_integrity());

  //
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>::Matrix out;

  //
  auto taskData = std::make_shared<ppc::core::TaskData>();
  krylov_m_matmul_strip_ha_vb_seq::fill_task_data(*taskData, lhs, rhs, out);

  //
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType> task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  EXPECT_EQ(out, ref);
}

TEST_F(krylov_m_matmul_strip_ha_vb_seq_test, bad_task_fail_validation) {
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>::Matrix out;

  krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> lhs{2, 5};
  krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> rhs{3, 2};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  krylov_m_matmul_strip_ha_vb_seq::fill_task_data(*taskData, lhs, rhs, out);

  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType> task(taskData);
  EXPECT_FALSE(task.validation());
}

TEST_F(krylov_m_matmul_strip_ha_vb_seq_test, bad_channels_fail_validation) {
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>::Matrix out;

  krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> lhs{2, 3};
  krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> rhs{3, 2};
  uint8_t trash{};

  // fill task data...
  auto taskData = std::make_shared<ppc::core::TaskData>();
  krylov_m_matmul_strip_ha_vb_seq::fill_task_data(*taskData, lhs, rhs, out);
  // ...and damage it
  taskData->outputs.emplace_back(&trash);

  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType> task(taskData);
  EXPECT_FALSE(task.validation());
}

TEST_F(krylov_m_matmul_strip_ha_vb_seq_test, bad_prealloc_fail_validation) {
  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>::Matrix out;

  krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> lhs{2, 3};
  krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> rhs{3, 2};

  // fill task data...
  auto taskData = std::make_shared<ppc::core::TaskData>();
  krylov_m_matmul_strip_ha_vb_seq::fill_task_data(*taskData, lhs, rhs, out);
  // ...and damage it
  taskData->outputs_count[0] += 2;

  krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType> task(taskData);
  EXPECT_FALSE(task.validation());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_matmul_strip_ha_vb_seq_test, krylov_m_matmul_strip_ha_vb_seq_test,
    ::testing::Values(
      MatTestParam(
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
      MatTestParam(
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
      MatTestParam(
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
