#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/milovankin_m_sum_of_vector_elements/include/ops_mpi.hpp"

namespace milovankin_m_sum_of_vector_elements_parallel {
[[nodiscard]] std::vector<int32_t> make_random_vector(int32_t size, int32_t val_min, int32_t val_max) {
  std::vector<int32_t> new_vector(size);

  for (int32_t i = 0; i < size; i++) {
    new_vector[i] = rand() % (val_max - val_min + 1) + val_min;
  }

  return new_vector;
}
}  // namespace milovankin_m_sum_of_vector_elements_parallel

void run_parallel_and_sequential_tasks(std::vector<int32_t> &input_vector, int64_t expected_sum) {
  boost::mpi::communicator world;
  std::vector<int64_t> result_parallel(1, 0);
  std::vector<int64_t> result_sequential(1, 0);

  // Task data parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(result_parallel.size());
  }

  // Parallel
  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Task data sequential
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    taskDataSeq->inputs_count.emplace_back(input_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(result_sequential.size());

    // Sequential
    milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    // Assert
    ASSERT_EQ(result_sequential[0], result_parallel[0]);
    ASSERT_EQ(result_sequential[0], expected_sum);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, randomVector50000) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;

  if (world.rank() == 0) {
    input_vector = milovankin_m_sum_of_vector_elements_parallel::make_random_vector(50000, -500, 5000);
  }

  run_parallel_and_sequential_tasks(input_vector, std::accumulate(input_vector.begin(), input_vector.end(), 0));
}

TEST(milovankin_m_sum_of_vector_elements_mpi, regularVector) {
  std::vector<int32_t> input_vector = {1, 2, 3, -5, 3, 43};
  run_parallel_and_sequential_tasks(input_vector, 47);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, positiveNumbers) {
  std::vector<int32_t> input_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  run_parallel_and_sequential_tasks(input_vector, 55);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, negativeNumbers) {
  std::vector<int32_t> input_vector = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  run_parallel_and_sequential_tasks(input_vector, -55);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, zeroVector) {
  std::vector<int32_t> input_vector = {0, 0, 0, 0, 0};
  run_parallel_and_sequential_tasks(input_vector, 0);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, tinyVector) {
  std::vector<int32_t> input_vector = {4, -20};
  run_parallel_and_sequential_tasks(input_vector, -16);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, emptyVector) {
  std::vector<int32_t> input_vector = {};
  run_parallel_and_sequential_tasks(input_vector, 0);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, validationNotPassed) {
  boost::mpi::communicator world;

  std::vector<int32_t> input = {1, 2, 3, -5};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    // Omitting output setup to cause validation to fail
  }

  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(vectorSumPar.validation());
  }
}
