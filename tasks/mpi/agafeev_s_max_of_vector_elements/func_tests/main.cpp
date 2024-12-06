#include <gtest/gtest.h>

#include <iostream>

#include "boost/mpi/communicator.hpp"
#include "mpi/agafeev_s_max_of_vector_elements/include/ops_mpi.hpp"
// #include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"

TEST(agafeev_s_max_of_vector_elements, test_find_in_100x100_matrix) {
  boost::mpi::communicator world;
  const int columns = 100;
  const int rows = 100;
  auto rand_gen = std::mt19937(1337);

  std::vector<int> in_matrix = agafeev_s_max_of_vector_elements_mpi::create_RandomMatrix<int>(rows, columns);
  std::vector<int> out(1, 0);
  const int right_answer = std::numeric_limits<int>::max();
  int index = rand_gen() % (rows * columns);
  in_matrix[index] = std::numeric_limits<int>::max();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(right_answer, out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_3x3_matrix) {
  boost::mpi::communicator world;
  const int columns = 3;
  const int rows = 3;
  auto rand_gen = std::mt19937(1337);

  std::vector<int> in_matrix = agafeev_s_max_of_vector_elements_mpi::create_RandomMatrix<int>(rows, columns);
  std::vector<int> out(1, 0);
  const int right_answer = std::numeric_limits<int>::max();
  int index = rand_gen() % (rows * columns);
  in_matrix[index] = std::numeric_limits<int>::max();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int> testTask(taskData);
  bool isValid = testTask.validation();
  ASSERT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(right_answer, out[0]);
}