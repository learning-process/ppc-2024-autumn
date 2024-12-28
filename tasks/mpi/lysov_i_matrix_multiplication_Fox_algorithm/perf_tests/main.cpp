// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_mpi.hpp"

static std::vector<double> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}
int matrix_size = 400;
std::vector<double> A = getRandomVector(matrix_size * matrix_size);
std::vector<double> B = getRandomVector(matrix_size * matrix_size);
TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  auto testMpiTaskParallel =
      std::make_shared<lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, taskrun) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}