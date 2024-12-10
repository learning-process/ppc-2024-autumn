#include <gtest/gtest.h>

#include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"

template <typename T>
std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 100;

  return matrix;
}

TEST(agafeev_s_max_of_vector_elements_seq, find_max_in_3x3_matrix) {
  const int rows = 3;
  const int columns = 3;
  auto rand_gen = std::mt19937(1337);

  // Create data

  std::vector<int> in_matrix = create_RandomMatrix<int>(rows, columns);
  std::vector<int> out(1, 0);
  const int right_answer = std::numeric_limits<int>::max();
  int index = rand_gen() % (rows * columns);
  in_matrix[index] = std::numeric_limits<int>::max();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
  taskData->inputs_count.emplace_back(in_matrix.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(right_answer, out[0]);
}

TEST(agafeev_s_max_of_vector_elements_seq, find_max_in_10x10_matrix) {
  const int rows = 10;
  const int columns = 10;
  auto rand_gen = std::mt19937(1337);

  // Create data

  std::vector<int> in_matrix = create_RandomMatrix<int>(rows, columns);
  std::vector<int> out(1, 0);
  const int right_answer = std::numeric_limits<int>::max();
  int index = rand_gen() % (rows * columns);
  in_matrix[index] = std::numeric_limits<int>::max();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
  taskData->inputs_count.emplace_back(in_matrix.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(right_answer, out[0]);
}

TEST(agafeev_s_max_of_vector_elements_seq, find_max_in_9x45_matrix) {
  const int rows = 10;
  const int columns = 10;
  auto rand_gen = std::mt19937(1337);

  // Create data

  std::vector<int> in_matrix = create_RandomMatrix<int>(rows, columns);
  std::vector<int> out(1, 0);
  const int right_answer = std::numeric_limits<int>::max();
  int index = rand_gen() % (rows * columns);
  in_matrix[index] = std::numeric_limits<int>::max();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
  taskData->inputs_count.emplace_back(in_matrix.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(right_answer, out[0]);
}

TEST(agafeev_s_max_of_vector_elements_seq, find_max_in_130x187_matrix) {
  const int rows = 10;
  const int columns = 10;
  auto rand_gen = std::mt19937(1337);

  // Create data

  std::vector<int> in_matrix = create_RandomMatrix<int>(rows, columns);
  std::vector<int> out(1, 0);
  const int right_answer = std::numeric_limits<int>::max();
  int index = rand_gen() % (rows * columns);
  in_matrix[index] = std::numeric_limits<int>::max();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
  taskData->inputs_count.emplace_back(in_matrix.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(right_answer, out[0]);
}

TEST(agafeev_s_max_of_vector_elements_seq, check_validate_func) {
  // Create data
  std::vector<int32_t> in(20, 1);
  std::vector<int32_t> out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, false);
}

TEST(agafeev_s_max_of_vector_elements_seq, check_wrong_order) {
  // Create data
  std::vector<float> in(20, 1);
  std::vector<float> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, true);
  testTask.pre_processing();
  ASSERT_ANY_THROW(testTask.post_processing());
}