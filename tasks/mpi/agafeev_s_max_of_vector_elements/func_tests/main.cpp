#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "mpi/agafeev_s_max_of_vector_elements/include/ops_mpi.hpp"

template <typename T>
void test_seq_and_mpi_versions(std::vector<T> &in_matrix, T right_answer) {}

TEST(agafeev_s_max_of_vector_elements, test_find_in_3x3_matrix) {
  boost::mpi::communicator world;
  // world.barrier();
  std::vector<int> in_matrix(9);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = agafeev_s_max_of_vector_elements_mpi::create_RandomMatrix<int>(3, 3);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  // world.barrier();
  testTask->post_processing();
  if (world.rank() == 0) {
    int right_answer = std::numeric_limits<int>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_100x100_matrix) {
  boost::mpi::communicator world;
  // world.barrier();
  std::vector<int> in_matrix(10000);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = agafeev_s_max_of_vector_elements_mpi::create_RandomMatrix<int>(100, 100);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  // world.barrier();
  testTask->post_processing();
  if (world.rank() == 0) {
    int right_answer = std::numeric_limits<int>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
  }
}