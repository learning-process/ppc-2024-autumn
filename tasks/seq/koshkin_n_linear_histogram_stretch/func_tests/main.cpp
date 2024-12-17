// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/koshkin_n_linear_histogram_stretch/include/ops_seq.hpp"

TEST(koshkin_n_linear_histogram_stretch_seq, test_correct_image) {
  const int count_size_vector = 6;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {15, 50, 45, 101, 92, 79};
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out = {0, 0, 0, 255, 252, 216}; // теоретически посчитано

  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(res_exp_out, out_vec);

}

TEST(koshkin_n_linear_histogram_stretch_seq, test_incorrect_rgb_size_image) {
  const int count_size_vector = 8;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {15, 50, 45, 101, 92, 79, 0, 0};
  std::vector<int> out_vec(count_size_vector, 0);


  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_NE(testTaskSequential.validation(), true);

}

TEST(koshkin_n_linear_histogram_stretch_seq, test_incorrect_value_color_range_image) {
  const int count_size_vector = 12;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {631, -2, 45, 101, 92, 79, 0, 0, 300, 255, 10, 15};
  std::vector<int> out_vec(count_size_vector, 0);

  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_empty_image) {
  const int count_size_vector = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {};
  std::vector<int> out_vec(count_size_vector, 0);

  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_EQ(testTaskSequential.validation(), false);
}