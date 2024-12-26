#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

std::vector<double> generateRandomMatrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  std::vector<double> matrix(size * size);
  for (int i = 0; i < size * size; i++) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

TEST(nasedkin_e_strassen_algorithm_perf_test, test_pipeline_run) {
  int matrixSize = 1024;
  boost::mpi::communicator world;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  auto testMpiTaskParallel = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer currentTimer;
  perfAttr->current_timer = [&] { return currentTimer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(nasedkin_e_strassen_algorithm_perf_test, test_task_run) {
  int matrixSize = 1024;
  boost::mpi::communicator world;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  auto testMpiTaskParallel = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer currentTimer;
  perfAttr->current_timer = [&] { return currentTimer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}