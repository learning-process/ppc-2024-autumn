// Copyright 2024 Sedova Olga
#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sedova_o_max_of_vector_elements/include/ops_seq.hpp"

std::vector<int> generate_random_vector(size_t size, size_t value) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = random() % (value + 1);
  }
  return vec;
}

std::vector<std::vector<int>> generate_random_matrix(size_t rows, size_t cols, size_t value) {
  std::vector<std::vector<int>> matrix(rows);
  for (size_t i = 0; i < rows; i++) {
    matrix[i] = generate_random_vector(cols, value);
  }
  return matrix;
}

TEST(sedova_o_max_of_vector_elements_seq, Test_Sum_Empty) {
  // Create data
  std::vector<int> in;
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  sedova_o_max_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sedova_o_max_of_vector_elements_seq, Test_Sum_Output_Incorrect) {
  const size_t count = 10;
  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(2, 0);  // Изменен размер выходного вектора
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  sedova_o_max_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sedova_o_max_of_vector_elements_seq, Test_Sum_Input_Incorrect) {
  const size_t count = 10;
  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(0);  // Неверный размер входного вектора
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  sedova_o_max_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

