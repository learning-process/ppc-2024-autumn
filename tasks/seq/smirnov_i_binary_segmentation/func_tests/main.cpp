#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/smirnov_i_binary_segmentation/include/ops_seq.hpp"

TEST(smirnov_i_binary_segmentation_seq, negative_sizes_img) {
  int cols = 1;
  int rows = -1;
  std::vector<int> img = {0};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), false);
}
TEST(smirnov_i_binary_segmentation_seq, not_enough_sizes_img) {
  int cols = 1;
  std::vector<int> img = {0};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(cols);
  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), false);
}
TEST(smirnov_i_binary_segmentation_seq, not_binary_img) {
  int cols = 1;
  int rows = 1;
  std::vector<int> img = {3};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);
  ASSERT_EQ(TestTaskSequential->validation(), false);
}
TEST(smirnov_i_binary_segmentation_seq, get_mask_for_scalar) {
  int cols = 1;
  int rows = 1;
  std::vector<int> img = {0};
  std::vector<int> expected_mask = {2};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> mask(rows * cols, 1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  taskDataSeq->outputs_count.emplace_back(cols);
  taskDataSeq->outputs_count.emplace_back(rows);

  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), true);
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();
  for (int i = 0; i < cols * rows; i++) {
    ASSERT_EQ(expected_mask[i], mask[i]);
  }
}
TEST(smirnov_i_binary_segmentation_seq, get_mask_small) {
  int cols = 2;
  int rows = 2;
  std::vector<int> img = {0, 255, 255, 255};
  std::vector<int> expected_mask = {2, 1, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> mask(rows * cols, 1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  taskDataSeq->outputs_count.emplace_back(cols);
  taskDataSeq->outputs_count.emplace_back(rows);

  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), true);
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();
  for (int i = 0; i < cols * rows; i++) {
    ASSERT_EQ(expected_mask[i], mask[i]);
  }
}

TEST(smirnov_i_binary_segmentation_seq, get_mask_medium) {
  int cols = 8;
  int rows = 4;
  std::vector<int> img = {0,   255, 0, 0,   255, 255, 0,   0,   255, 255, 255, 0, 255, 255, 0,   255,
                          255, 0,   0, 255, 255, 255, 255, 255, 255, 0,   255, 0, 255, 0,   255, 255};
  std::vector<int> expected_mask = {2, 1, 3, 3, 1, 1, 4, 4, 1, 1, 1, 3, 1, 1, 4, 1,
                                    1, 5, 5, 1, 1, 1, 1, 1, 1, 5, 1, 5, 1, 6, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> mask(rows * cols, 1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  taskDataSeq->outputs_count.emplace_back(cols);
  taskDataSeq->outputs_count.emplace_back(rows);

  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), true);
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();
  for (int i = 0; i < cols * rows; i++) {
    ASSERT_EQ(expected_mask[i], mask[i]);
  }
}

TEST(smirnov_i_binary_segmentation_seq, get_mask_large) {
  int cols = 20;
  int rows = 21;
  std::vector<int> img = {
      0,   255, 255, 0,   0,   0,   0,   0,   255, 255, 0,   255, 0,   255, 0,   0,   255, 0,   0,   0,   0,   255, 255,
      0,   0,   0,   255, 255, 0,   0,   255, 255, 0,   255, 255, 255, 0,   255, 255, 255, 0,   0,   0,   255, 255, 255,
      255, 255, 0,   0,   0,   0,   255, 255, 255, 255, 0,   0,   0,   255, 255, 255, 255, 0,   0,   255, 0,   0,   255,
      0,   0,   0,   0,   0,   0,   255, 0,   0,   255, 0,   0,   0,   255, 0,   0,   255, 255, 0,   0,   0,   0,   0,
      255, 255, 255, 0,   255, 0,   0,   255, 0,   0,   255, 0,   0,   0,   0,   255, 0,   255, 0,   255, 0,   0,   255,
      255, 0,   255, 0,   0,   255, 255, 255, 255, 255, 255, 0,   255, 0,   0,   0,   0,   255, 255, 255, 255, 255, 0,
      255, 0,   255, 0,   0,   0,   0,   255, 0,   0,   255, 255, 0,   255, 0,   255, 255, 0,   255, 255, 0,   0,   255,
      255, 255, 0,   0,   255, 0,   255, 255, 0,   255, 255, 0,   0,   255, 0,   0,   0,   0,   0,   0,   255, 0,   0,
      255, 0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   255, 255, 0,   255, 0,   255, 0,   0,   0,   255, 0,   255,
      255, 0,   255, 0,   255, 255, 0,   0,   255, 0,   0,   0,   0,   255, 0,   0,   255, 0,   255, 255, 255, 255, 255,
      0,   255, 255, 0,   0,   0,   0,   255, 255, 0,   255, 255, 0,   255, 255, 255, 0,   0,   0,   255, 255, 0,   255,
      255, 255, 0,   0,   0,   255, 0,   255, 255, 0,   255, 0,   255, 0,   255, 255, 255, 0,   255, 0,   255, 255, 0,
      0,   0,   255, 0,   255, 255, 0,   0,   255, 0,   255, 0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   255, 255,
      0,   255, 255, 0,   255, 0,   0,   255, 0,   0,   0,   255, 255, 255, 255, 255, 0,   0,   255, 255, 0,   0,   0,
      255, 0,   0,   255, 255, 255, 255, 0,   0,   0,   255, 0,   255, 255, 0,   255, 0,   0,   255, 0,   0,   255, 255,
      0,   255, 255, 0,   0,   255, 255, 255, 255, 255, 0,   255, 255, 0,   255, 255, 0,   255, 0,   0,   0,   255, 255,
      0,   255, 0,   0,   255, 0,   0,   255, 255, 255, 0,   0,   255, 255, 0,   255, 0,   255, 0,   255, 255, 0,   255,
      0,   255, 0,   0,   255, 255, 0,   255, 255, 255, 0,   255, 255, 0,   255, 0,   0,   255, 255, 0,   255, 255, 0,
      0,   0,   0,   255, 0,   0};
  std::vector<int> expected_mask = {
      2,  1,  1, 3,  3,  3, 3,  3,  1,  1,  4,  1,  5, 1,  3,  3, 1, 6,  6,  6,  2,  1,  1, 3,  3,  3,  1,  1, 3,  3,
      1,  1,  5, 1,  1,  1, 3,  1,  1,  1,  2,  2,  2, 1,  1,  1, 1, 1,  3,  3,  3,  3,  1, 1,  1,  1,  3,  3, 3,  1,
      1,  1,  1, 2,  2,  1, 3,  3,  1,  3,  3,  3,  3, 3,  3,  1, 3, 3,  1,  3,  7,  7,  1, 2,  2,  1,  1,  3, 3,  3,
      3,  3,  1, 1,  1,  3, 1,  3,  3,  1,  7,  7,  1, 2,  2,  2, 2, 1,  3,  1,  3,  1,  3, 3,  1,  1,  3,  1, 3,  3,
      1,  1,  1, 1,  1,  1, 2,  1,  3,  3,  3,  3,  1, 1,  1,  1, 1, 3,  1,  3,  1,  2,  2, 2,  2,  1,  2,  2, 1,  1,
      3,  1,  3, 1,  1,  3, 1,  1,  3,  3,  1,  1,  1, 2,  2,  1, 2, 1,  1,  8,  1,  1,  3, 3,  1,  3,  3,  3, 3,  3,
      2,  1,  2, 2,  1,  2, 2,  2,  2,  1,  1,  3,  3, 3,  3,  1, 1, 3,  1,  3,  1,  2,  2, 2,  1,  2,  1,  1, 2,  1,
      9,  1,  1, 3,  3,  1, 3,  3,  3,  3,  1,  2,  2, 1,  2,  1, 1, 1,  1,  1,  9,  1,  1, 3,  3,  3,  3,  1, 1,  3,
      1,  1,  2, 1,  1,  1, 10, 10, 10, 1,  1,  9,  1, 1,  1,  3, 3, 3,  1,  3,  1,  1,  2, 1,  2,  1,  10, 1, 1,  1,
      10, 1,  9, 1,  1,  3, 3,  3,  1,  3,  1,  1,  2, 2,  1,  2, 1, 10, 10, 10, 10, 10, 1, 1,  3,  3,  3,  1, 1,  3,
      1,  1,  2, 1,  2,  2, 1,  10, 10, 10, 1,  1,  1, 1,  1,  3, 3, 1,  1,  3,  2,  2,  1, 2,  2,  1,  1,  1, 1,  10,
      10, 10, 1, 11, 1,  1, 3,  1,  3,  3,  1,  2,  2, 1,  1,  2, 1, 1,  10, 10, 1,  1,  1, 1,  1,  12, 1,  1, 3,  1,
      1,  2,  1, 2,  2,  2, 1,  1,  10, 1,  10, 10, 1, 13, 13, 1, 1, 1,  3,  3,  1,  1,  2, 1,  2,  1,  2,  1, 1,  10,
      1,  10, 1, 13, 13, 1, 1,  14, 1,  1,  1,  15, 1, 1,  2,  1, 2, 2,  1,  1,  10, 1,  1, 13, 13, 13, 13, 1, 14, 14};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> mask(rows * cols, 1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  taskDataSeq->outputs_count.emplace_back(cols);
  taskDataSeq->outputs_count.emplace_back(rows);

  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), true);
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();
  for (int i = 0; i < cols * rows; i++) {
    ASSERT_EQ(expected_mask[i], mask[i]);
  }
}