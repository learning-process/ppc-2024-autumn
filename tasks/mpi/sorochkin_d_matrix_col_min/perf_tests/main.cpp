#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"

static std::vector<double> randv(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(sz);
  for (size_t i = 0; i < sz; i++) {
    vec[i] = -50 + (gen() / 10.);
  }
  return vec;
}

TEST(sorochkin_d_matrix_col_min_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  const int rows = 5000;
  const int cols = 5000;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = randv(rows * cols);
    out.resize(cols);
    // in
    taskDataPar->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataPar->inputs_count = {rows, cols};
    // out
    taskDataPar->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataPar->outputs_count = {cols};
  }

  // Create Task
  auto testTaskParallel = std::make_shared<sorochkin_d_matrix_col_min_mpi::TestMPITaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(sorochkin_d_matrix_col_min_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  const int rows = 5000;
  const int cols = 5000;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = randv(rows * cols);
    out.resize(cols);
    // in
    taskDataPar->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataPar->inputs_count = {rows, cols};
    // out
    taskDataPar->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataPar->outputs_count = {cols};
  }

  // Create Task
  auto testTaskParallel = std::make_shared<sorochkin_d_matrix_col_min_mpi::TestMPITaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
