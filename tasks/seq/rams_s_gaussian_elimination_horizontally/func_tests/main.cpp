#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/rams_s_gaussian_elimination_horizontally/include/ops_seq.hpp"

void rams_s_gaussian_elimination_horizontally_seq_run_test(std::vector<double> &&in, std::vector<double> &&expected) {
  std::vector<double> out(expected.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  rams_s_gaussian_elimination_horizontally_seq::TaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < expected.size(); i++) {
    if (std::isnan(expected[i])) {
      ASSERT_TRUE(std::isnan(out[i]));
    } else {
      ASSERT_DOUBLE_EQ(expected[i], out[i]);
    }
  }
}

#define TEST_IT(case_name, ...)                                         \
  TEST(rams_s_gaussian_elimination_horizontally_seq, case_name) {       \
    rams_s_gaussian_elimination_horizontally_seq_run_test(__VA_ARGS__); \
  }

// clang-format off
TEST_IT(0, {}, {})

TEST_IT(1, {
     0,  0,  3,  6,
     0,  4,  0, -2,
     7,  0,  0,  0
  }, {0,0.5,-2})

TEST_IT(2, {
     0,  0,  3,  6,
     0,  2,  0, -4,
     7,  0,  0,  1,
     0,  0,  6, 12,
  }, {-1.0/7,2,-2})

TEST_IT(3, {
     7,  1,  2, 12,
     0,  5,  0,  5,
     0,  0,  4,  8,
     0,  0,  2,  4
  }, {-1, -1, -2})
TEST_IT(4, {
     2,  0,  2,  2, -6,
     0,  0,  2,  2, -4,
     0,  0,  0,  4, -4,
     0,  3,  0,  0,  3,
  }, {1, -1, 1, 1})
// clang-format on

void rams_s_gaussian_elimination_horizontally_seq_run_validation_test(std::vector<double> &&in, int outputs_count,
                                                                      bool expected) {
  std::vector<double> out(outputs_count, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  rams_s_gaussian_elimination_horizontally_seq::TaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(expected, testTaskSequential.validation());
}
#define TEST_VALIDATION(case_name, ...)                                            \
  TEST(rams_s_gaussian_elimination_horizontally_seq, case_name) {                  \
    rams_s_gaussian_elimination_horizontally_seq_run_validation_test(__VA_ARGS__); \
  }
// clang-format off
TEST_VALIDATION(rejects_when_input_is_not_a_matrix_with_specified_cols_count, {
  1, 1, -2,
  2, 2
}, 2, false)
TEST_VALIDATION(rejects_when_input_is_not_a_matrix_with_specified_cols_count_2, {
  1, 1, -2,
  2, 2, -4,
}, 3, false)
TEST_VALIDATION(rejects_if_matrix_is_not_singular, {
  1, 1, 1, -3,
  2, 2, 2, -6,
}, 3, false)
TEST_VALIDATION(rejects_if_matrix_is_not_singular2, {
  1, 0, 1, -2,
  0, 0, 0, -5,
  0, 0, 1, -1,
}, 3, false)
TEST_VALIDATION(rejects_if_matrix_is_not_singular3, {
  1, 0, 1, 1, -3,
  0, 0, 0, 1, -5,
  0, 0, 1, 1, -6,
}, 4, false)
// clang-format on
