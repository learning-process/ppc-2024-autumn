#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "../include/ops_seq.hpp"

static void mcm_test(std::vector<int> &&in, uint32_t rows, uint32_t cols, const std::vector<int> &ref) {
  // Create data
  std::vector<int> out(cols);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // in
  taskDataSeq->inputs = {reinterpret_cast<uint8_t *>(in.data())};
  taskDataSeq->inputs_count = {rows, cols};
  // out
  taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  taskDataSeq->outputs_count = {cols};

  // Create Task
  sorochkin_d_matrix_col_min_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(out, ref);
}

TEST(sorochkin_d_matrix_col_min_seq, Test_1_1) { mcm_test({5}, 1, 1, {5}); }
TEST(sorochkin_d_matrix_col_min_seq, Test_1_2) { mcm_test({2, 1}, 1, 2, {2, 1}); }
TEST(sorochkin_d_matrix_col_min_seq, Test_2_1) { mcm_test({2, 1}, 2, 1, {1}); }
TEST(sorochkin_d_matrix_col_min_seq, Test_9_16) {
  // clang-format off
  mcm_test({
    11, 12, 13, 14, 15, 16, 17, 18, 19, 110, 111, 112, 113, 114, 115, 116,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 410, 411, 412, 413, 414, 415, 416,
     1,  2 , 3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 510, 511, 512, 513, 514, 515, 516,
    61, 62, 63, 64, -65, 66, 67, 68, 69, 610, 611, 612, 613, 614, 615, 616,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 710, 711, 712, 713, 714, 715, 716,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 810, 811, 812, 813, 814, 815, 816,
  }, 9, 16, {1,  2 , 3,  4,  -65,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16});
  // clang-format on
}