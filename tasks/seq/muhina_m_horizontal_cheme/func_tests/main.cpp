// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/muhina_m_horizontal_cheme/include/ops_seq.hpp"

TEST(muhina_m_horizontal_cheme_seq, Test_Validation_1) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> vec = {};
  std::vector<int> result(0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential task(taskDataSeq);

  ASSERT_EQ(task.validation(), false);
}

TEST(muhina_m_horizontal_cheme_seq, Test_Validation_2) {
  std::vector<int> input_matrix = {};
  std::vector<int> input_vector = {1, 2};
  std::vector<int> result(0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), false);
}

TEST(muhina_m_horizontal_cheme_seq, Test_Matrix_Vector_Multiplication_1) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> vec = {1, 2, 3};
  std::vector<int> output_result(2);
  std::vector<int> expected_result = {14, 32};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential task(taskDataSeq);

  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_EQ(output_result, expected_result);
}

TEST(muhina_m_horizontal_cheme_seq, Test_Matrix_Vector_Multiplication_2) {
  std::vector<int> matrix = {-1, -2, -3, -4, -5, -6};
  std::vector<int> vec = {-1, -2, -3};
  std::vector<int> output_result(2);
  std::vector<int> expected_result = {14, 32};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential task(taskDataSeq);

  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_EQ(output_result, expected_result);
}

TEST(muhina_m_horizontal_cheme_seq, Test_Matrix_Vector_Multiplication_3) {
  std::vector<int> matrix = {1, 2, 3};
  std::vector<int> vec = {1, 2, 3};
  std::vector<int> output_result(1);
  std::vector<int> expected_result = {14};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential task(taskDataSeq);

  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_EQ(output_result, expected_result);
}
