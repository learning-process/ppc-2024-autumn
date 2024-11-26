#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/borisov_s_broadcast/include/ops_mpi.hpp"

std::vector<double> borisov_s_broadcast::getRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);

  for (auto &val : matrix) {
    val = dis(gen);
  }

  return matrix;
}

TEST(parallel_clustering_perf_test, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t rows = 1000;
  const size_t cols = 1000;
  double epsilon = 1.0;

  std::vector<double> global_matrix;
  std::vector<int> global_result(rows, 0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = borisov_s_broadcast::getRandomMatrix(rows, cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testTaskParallel = std::make_shared<borisov_s_broadcast::MPITaskMatrixClustering>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_result.size(), rows);
  }
}

TEST(parallel_clustering_perf_test, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t rows = 1000;
  const size_t cols = 1000;
  double epsilon = 1.0;

  std::vector<double> global_matrix;
  std::vector<int> global_result(rows, 0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = borisov_s_broadcast::getRandomMatrix(rows, cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testTaskParallel = std::make_shared<borisov_s_broadcast::MPITaskMatrixClustering>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_result.size(), rows);
  }
}