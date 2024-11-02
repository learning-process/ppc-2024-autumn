// mpi/func_tests.cpp
#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <limits>
#include <numeric>
#include <vector>

#include "mpi/anufriev_d_max_of_vector_elements/include/ops_mpi_anufriev.hpp"

void run_parallel_and_sequential_tasks(std::vector<int32_t>& input_vector, int32_t expected_max) {
  boost::mpi::communicator world;
  int32_t result_parallel = std::numeric_limits<int32_t>::min();
  int32_t result_sequential = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_parallel));
    taskDataPar->outputs_count.emplace_back(1);
  }

  anufriev_d_max_of_vector_elements_parallel::VectorMaxPar testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataSeq->inputs_count.emplace_back(input_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_sequential));
    taskDataSeq->outputs_count.emplace_back(1);

    anufriev_d_max_of_vector_elements_parallel::VectorMaxSeq testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(result_sequential, result_parallel);
    ASSERT_EQ(result_sequential, expected_max);
  }
}

TEST(anufriev_d_max_of_vector_elements_mpi, randomVector50000) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;

  if (world.rank() == 0) {
    input_vector = anufriev_d_max_of_vector_elements_parallel::make_random_vector(50000, -500, 5000);
  }
  boost::mpi::broadcast(world, input_vector, 0);  

  int32_t expected_max = std::numeric_limits<int32_t>::min();
  if (world.rank() == 0) {
    expected_max = *std::max_element(input_vector.begin(), input_vector.end());
  }

  run_parallel_and_sequential_tasks(input_vector, expected_max);
}

TEST(anufriev_d_max_of_vector_elements_mpi, regularVector) {
  std::vector<int32_t> input_vector = {1, 2, 3, -5, 3, 43};
  run_parallel_and_sequential_tasks(input_vector, 43);
}

TEST(anufriev_d_max_of_vector_elements_mpi, positiveNumbers) {
  std::vector<int32_t> input_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  run_parallel_and_sequential_tasks(input_vector, 10);
}

TEST(anufriev_d_max_of_vector_elements_mpi, negativeNumbers) {
  std::vector<int32_t> input_vector = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  run_parallel_and_sequential_tasks(input_vector, -1);
}

TEST(anufriev_d_max_of_vector_elements_mpi, zeroVector) {
  std::vector<int32_t> input_vector = {0, 0, 0, 0, 0};
  run_parallel_and_sequential_tasks(input_vector, 0);
}

TEST(anufriev_d_max_of_vector_elements_mpi, tinyVector) {
  std::vector<int32_t> input_vector = {4, -20};
  run_parallel_and_sequential_tasks(input_vector, 4);
}

TEST(anufriev_d_max_of_vector_elements_mpi, emptyVector) {
  std::vector<int32_t> input_vector = {};
  run_parallel_and_sequential_tasks(input_vector, std::numeric_limits<int32_t>::min());
}

TEST(anufriev_d_max_of_vector_elements_mpi, validationNotPassed) {
  boost::mpi::communicator world;
  std::vector<int32_t> input = {1, 2, 3, -5};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  }

  anufriev_d_max_of_vector_elements_parallel::VectorMaxPar vectorMaxPar(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(vectorMaxPar.validation());
  }
}