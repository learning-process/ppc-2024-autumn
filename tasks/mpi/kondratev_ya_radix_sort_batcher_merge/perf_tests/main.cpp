// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kondratev_ya_radix_sort_batcher_merge/include/ops_mpi.hpp"
namespace kondratev_ya_radix_sort_batcher_merge_mpi {
std::vector<double> getRandomVector(uint32_t size) {
  std::srand(std::time(nullptr));
  std::vector<double> vec(size);

  double lower_bound = -10000;
  double upper_bound = 10000;
  for (uint32_t i = 0; i < size; i++) {
    vec[i] = lower_bound + std::rand() / (double)RAND_MAX * (upper_bound - lower_bound);
  }
  return vec;
}
}  // namespace kondratev_ya_radix_sort_batcher_merge_mpi

TEST(kondratev_ya_radix_sort_batcher_merge_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 5000000;
    in = kondratev_ya_radix_sort_batcher_merge_mpi::getRandomVector(size);
    out.resize(size);

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel = std::make_shared<kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel>(taskDataMPI);
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 5000000;
    in = kondratev_ya_radix_sort_batcher_merge_mpi::getRandomVector(size);
    out.resize(size);

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel = std::make_shared<kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel>(taskDataMPI);
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
