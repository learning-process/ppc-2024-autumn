// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>

#include "seq/petrov_a_ribbon_vertical_scheme/include/ops_seq.hpp"

TEST(petrov_a_ribbon_vertical_scheme_seq, Test_MatrixVector) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> vector = {1, 1, 1};
  std::vector<int> result(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs_count.push_back(2);
  taskData->inputs_count.push_back(3);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskData->inputs_count.push_back(vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.push_back(result.size());

  petrov_a_ribbon_vertical_scheme_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  ASSERT_TRUE(task.run());
  task.post_processing();

  ASSERT_EQ(result[0], 6);
  ASSERT_EQ(result[1], 15);
}
TEST(petrov_a_ribbon_vertical_scheme_seq, Test_func) {

  const int rows = 3;
  const int cols = 3;

  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::vector<int> vector = {1, 1, 1};
  std::vector<int> result(rows, 0);

  std::vector<int> expected_result = {6, 15, 24};


  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(rows);


  auto testTaskSeq = std::make_shared<petrov_a_ribbon_vertical_scheme_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_EQ(testTaskSeq->validation(), true);
  testTaskSeq->pre_processing();
  ASSERT_TRUE(testTaskSeq->run());
  testTaskSeq->post_processing();


  for (int i = 0; i < rows; ++i) {
    ASSERT_EQ(result[i], expected_result[i]) << "Mismatch at row " << i;
  }

  ASSERT_FALSE(result.empty());
}
TEST(petrov_a_ribbon_vertical_scheme_seq, Test_negativ) {
  const int rows = 3;
  const int cols = 3;

  std::vector<int> matrix = {-1, 2, -3, 4, -5, 6, -7, 8, -9};

  std::vector<int> vector = {1, -1, 2};
  std::vector<int> result(rows, 0);

  std::vector<int> expected_result = {
      (-1) * 1 + 2 * (-1) + (-3) * 2,
      4 * 1 + (-5) * (-1) + 6 * 2,
      (-7) * 1 + 8 * (-1) + (-9) * 2
  };

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(rows);

  // Создание Task
  auto testTaskSeq = std::make_shared<petrov_a_ribbon_vertical_scheme_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_EQ(testTaskSeq->validation(), true);
  testTaskSeq->pre_processing();
  ASSERT_TRUE(testTaskSeq->run());
  testTaskSeq->post_processing();

  for (int i = 0; i < rows; ++i) {
    ASSERT_EQ(result[i], expected_result[i]) << "Mismatch at row " << i;  // Проверка результата
  }

  ASSERT_FALSE(result.empty());
}
TEST(petrov_a_ribbon_vertical_scheme_seq, Test_zeros) {
  const int rows = 3;
  const int cols = 3;

  std::vector<int> matrix(rows * cols, 0);
  std::vector<int> vector(cols, 0);
  std::vector<int> result(rows, 0);

  std::vector<int> expected_result(rows, 0);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(rows);

  // Создание Task
  auto testTaskSeq = std::make_shared<petrov_a_ribbon_vertical_scheme_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_EQ(testTaskSeq->validation(), true);
  testTaskSeq->pre_processing();
  ASSERT_TRUE(testTaskSeq->run());
  testTaskSeq->post_processing();

  for (int i = 0; i < rows; ++i) {
    ASSERT_EQ(result[i], expected_result[i]) << "Mismatch at row " << i;
  }

  ASSERT_FALSE(result.empty());
}