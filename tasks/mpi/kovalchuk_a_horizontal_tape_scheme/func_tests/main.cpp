#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kovalchuk_a_horizontal_tape_scheme/include/ops_mpi.hpp"

std::vector<int> getRandomVectora(int sz, int min = -999, int max = 999);
std::vector<std::vector<int>> getRandomMatrixa(int rows, int columns, int min = -999, int max = 999);

std::vector<int> getRandomVectora(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrixa(int rows, int columns, int min, int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVectora(columns, min, max);
  }
  return vec;
}

TEST(kovalchuk_a_horizontal_tape_scheme_mpi, Test_Matrix_10_10) {
  const int count_rows = 10;
  const int count_columns = 10;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = getRandomMatrixa(count_rows, count_columns);
    global_vector = getRandomVectora(count_columns);

    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_result(count_rows, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    kovalchuk_a_horizontal_tape_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_result, global_result);
  }
}