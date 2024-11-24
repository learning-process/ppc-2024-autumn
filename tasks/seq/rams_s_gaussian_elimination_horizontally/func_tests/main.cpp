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

const double NaN = std::numeric_limits<double>::quiet_NaN();

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
     0,  1, -2,  0,
    -1, -2,  1, -1,
     2,  3,  0,  2
  }, {NaN, NaN, NaN})

TEST_IT(4, {
     0,  1, -2,  0,
     0,  2, -4,  0,
     2,  0,  0, -4
  }, {2, NaN, NaN})

TEST_IT(5, {
     7,  0,  2, 11,
     0,  0,  4,  8,
     0,  0,  2,  4
  }, {-1, NaN, -2})

TEST_IT(6, {
     7,  1,  2, 12,
     0,  0,  4,  8,
     0,  0,  2,  4
  }, {NaN, NaN, -2})

TEST_IT(7, {
     7,  1,  2, 12,
     0,  5,  0,  5,
     0,  0,  4,  8,
     0,  0,  2,  4
  }, {-1, -1, -2})

TEST_IT(8, {
     7,  1,  2, 12,
  }, {NaN, NaN, NaN})
// clang-format on
