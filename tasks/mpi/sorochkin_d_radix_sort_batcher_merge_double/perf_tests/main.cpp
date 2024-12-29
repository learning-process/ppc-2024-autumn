#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
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

TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = randv(128'000);
    out.resize(in.size());
    // in
    taskDataPar->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataPar->inputs_count = {static_cast<uint32_t>(in.size())};
    // out
    taskDataPar->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataPar->outputs_count = {static_cast<uint32_t>(out.size())};
  }

  // Create Task
  auto testTaskParallel =
      std::make_shared<sorochkin_d_radix_sort_batcher_merge_double_mpi::TestMPITaskParallel>(taskDataPar);

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

TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = randv(128'000);
    out.resize(in.size());
    // in
    taskDataPar->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataPar->inputs_count = {static_cast<uint32_t>(in.size())};
    // out
    taskDataPar->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataPar->outputs_count = {static_cast<uint32_t>(out.size())};
  }

  // Create Task
  auto testTaskParallel =
      std::make_shared<sorochkin_d_radix_sort_batcher_merge_double_mpi::TestMPITaskParallel>(taskDataPar);

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
