#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, test_pipeline_run) {
  mpi::environment env;
  mpi::communicator world;

  int N = 1000000;
  std::vector<double> inputData(N);
  std::vector<double> xPar(N, 0.0);

  if (world.rank() == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);

    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  auto parallelRadixSort =
      std::make_shared<sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel>(
          taskDataPar);

  ASSERT_TRUE(parallelRadixSort->validation()) << "Validation failed!";
  parallelRadixSort->pre_processing();
  parallelRadixSort->run();
  parallelRadixSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelRadixSort);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, test_task_run) {
  mpi::environment env;
  mpi::communicator world;

  int N = 1000000;
  std::vector<double> inputData;
  std::vector<double> xPar(N, 0.0);

  if (world.rank() == 0) {
    inputData.resize(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);

    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  auto parallelRadixSort =
      std::make_shared<sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel>(
          taskDataPar);

  ASSERT_TRUE(parallelRadixSort->validation()) << "Validation failed!";
  parallelRadixSort->pre_processing();
  parallelRadixSort->run();
  parallelRadixSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelRadixSort);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
