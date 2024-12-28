// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/pikarychev_i_monte_carlo/include/ops_mpi.hpp"

TEST(pikarychev_i_monte_carlo_mpi, SequentialVsParallel) {
  const double a = 0.0;
  const double b = 1.0;
  const int num_samples = 100000;
  const int seed = 12345;

  std::vector<double> sequential_res(1, 0.0);
  std::vector<double> parallel_res(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> inputs = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataSeq->inputs_count.emplace_back(inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_res.data()));
  taskDataSeq->outputs_count.emplace_back(sequential_res.size());

  pikarychev_i_monte_carlo_parallel::TestMPITaskSequential sequentialTask(taskDataSeq);
  ASSERT_TRUE(sequentialTask.validation());
  ASSERT_TRUE(sequentialTask.pre_processing());
  ASSERT_TRUE(sequentialTask.run());
  ASSERT_TRUE(sequentialTask.post_processing());

  double sequential_result = sequential_res[0];

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataPar->inputs_count.emplace_back(inputs.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_res.data()));
  taskDataPar->outputs_count.emplace_back(parallel_res.size());

  pikarychev_i_monte_carlo_parallel::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  ASSERT_TRUE(parallelTask.pre_processing());
  ASSERT_TRUE(parallelTask.run());
  ASSERT_TRUE(parallelTask.post_processing());

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    double parallel_result = parallel_res[0];
    ASSERT_NEAR(sequential_result, parallel_result, 0.1);
  }
}

TEST(pikarychev_i_monte_carlo_mpi, SequentialVsParallel_1) {
  const double a = 0.0;
  const double b = 0.0;
  const int num_samples = 1;
  const int seed = 12345;

  std::vector<double> sequential_res(1, 0.0);
  std::vector<double> parallel_res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> inputs = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataSeq->inputs_count.emplace_back(inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_res.data()));
  taskDataSeq->outputs_count.emplace_back(sequential_res.size());

  pikarychev_i_monte_carlo_parallel::TestMPITaskSequential sequentialTask(taskDataSeq);
  ASSERT_TRUE(sequentialTask.validation());
  ASSERT_TRUE(sequentialTask.pre_processing());
  ASSERT_TRUE(sequentialTask.run());
  ASSERT_TRUE(sequentialTask.post_processing());
  double sequential_result = sequential_res[0];
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataPar->inputs_count.emplace_back(inputs.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_res.data()));
  taskDataPar->outputs_count.emplace_back(parallel_res.size());
  pikarychev_i_monte_carlo_parallel::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  ASSERT_TRUE(parallelTask.pre_processing());
  ASSERT_TRUE(parallelTask.run());
  ASSERT_TRUE(parallelTask.post_processing());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    double parallel_result = parallel_res[0];
    ASSERT_EQ(sequential_result, parallel_result);
  }
}

TEST(pikarychev_i_monte_carlo_mpi, SequentialVsParallel_4) {
  const double a = -1.0;
  const double b = 1.0;
  const int num_samples = 100;
  const int seed = 1;
  std::vector<double> sequential_res(1, 0.33335967263763133);
  std::vector<double> parallel_res(1, 0.33335967263763133);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> inputs = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataSeq->inputs_count.emplace_back(inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_res.data()));
  taskDataSeq->outputs_count.emplace_back(sequential_res.size());
  pikarychev_i_monte_carlo_parallel::TestMPITaskSequential sequentialTask(taskDataSeq);
  ASSERT_TRUE(sequentialTask.validation());
  ASSERT_TRUE(sequentialTask.pre_processing());
  ASSERT_TRUE(sequentialTask.run());
  ASSERT_TRUE(sequentialTask.post_processing());
  double sequential_result = sequential_res[0];
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataPar->inputs_count.emplace_back(inputs.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_res.data()));
  taskDataPar->outputs_count.emplace_back(parallel_res.size());
  pikarychev_i_monte_carlo_parallel::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  ASSERT_TRUE(parallelTask.pre_processing());
  ASSERT_TRUE(parallelTask.run());
  ASSERT_TRUE(parallelTask.post_processing());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    double parallel_result = parallel_res[0];
    ASSERT_NEAR(sequential_result, parallel_result, 0.1);
  }
}

TEST(pikarychev_i_monte_carlo_mpi, SequentialVsParallel_5) {
  const double a = -1.0;
  const double b = -2.0;
  const int num_samples = 100;
  const int seed = 1;
  std::vector<double> sequential_res(1, 0.33335967263763133);
  std::vector<double> parallel_res(1, 0.33335967263763133);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> inputs = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataSeq->inputs_count.emplace_back(inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_res.data()));
  taskDataSeq->outputs_count.emplace_back(sequential_res.size());
  pikarychev_i_monte_carlo_parallel::TestMPITaskSequential sequentialTask(taskDataSeq);
  ASSERT_TRUE(sequentialTask.validation());
  ASSERT_TRUE(sequentialTask.pre_processing());
  ASSERT_TRUE(sequentialTask.run());
  ASSERT_TRUE(sequentialTask.post_processing());
  double sequential_result = sequential_res[0];
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataPar->inputs_count.emplace_back(inputs.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_res.data()));
  taskDataPar->outputs_count.emplace_back(parallel_res.size());
  pikarychev_i_monte_carlo_parallel::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  ASSERT_TRUE(parallelTask.pre_processing());
  ASSERT_TRUE(parallelTask.run());
  ASSERT_TRUE(parallelTask.post_processing());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    double parallel_result = parallel_res[0];
    ASSERT_NEAR(sequential_result, parallel_result, 0.1);
  }
}

TEST(pikarychev_i_monte_carlo_mpi, SequentialVsParallel_reg9) {
  std::random_device dev;
  std::mt19937 gen(dev());
  double a = (gen() % 100) / 100.0;
  double b = (gen() % 100) / 100.0;
  const int num_samples = 1000;
  const int seed = 1;
  std::vector<double> sequential_res(1, 0.0);
  std::vector<double> parallel_res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<double> inputs = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataSeq->inputs_count.emplace_back(inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_res.data()));
  taskDataSeq->outputs_count.emplace_back(sequential_res.size());
  pikarychev_i_monte_carlo_parallel::TestMPITaskSequential sequentialTask(taskDataSeq);
  ASSERT_TRUE(sequentialTask.validation());
  ASSERT_TRUE(sequentialTask.pre_processing());
  ASSERT_TRUE(sequentialTask.run());
  ASSERT_TRUE(sequentialTask.post_processing());
  double sequential_result = sequential_res[0];
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  taskDataPar->inputs_count.emplace_back(inputs.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_res.data()));
  taskDataPar->outputs_count.emplace_back(parallel_res.size());
  pikarychev_i_monte_carlo_parallel::TestMPITaskParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  ASSERT_TRUE(parallelTask.pre_processing());
  ASSERT_TRUE(parallelTask.run());
  ASSERT_TRUE(parallelTask.post_processing());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    double parallel_result = parallel_res[0];

    ASSERT_NEAR(sequential_result, parallel_result, 0.1);
  }
}
