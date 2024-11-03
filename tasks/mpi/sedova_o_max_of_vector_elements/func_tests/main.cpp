#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sedova_o_max_of_vector_elements/include/ops_mpi.hpp"

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
TEST(Parallel_Operations_MPI, Test1) { ASSERT_NO_THROW(generate_random_vector(10, 10)); }
TEST(Parallel_Operations_MPI, Test2) { ASSERT_NO_THROW(generate_random_matrix(10, 10, 10)); }
TEST(Parallel_Operations_MPI, Test3) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -30);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matr = generate_random_matrix(0, 0, 20);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
    sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
TEST(Parallel_Operations_MPI, Test4) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -30);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matr = generate_random_matrix(1, 1, 20);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
    sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }
}
TEST(Parallel_Operations_MPI, Test5) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -20);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(1, 5, 20);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(5);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);
  ASSERT_EQ(testMpiTaskParallel.run(), true);
  ASSERT_EQ(testMpiTaskParallel.post_processing(), true);
}
