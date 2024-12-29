// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/veliev_e_sobel_operator/include/ops_mpi.hpp"

namespace veliev_e_sobel_operator_mpi_p {
std::vector<double> create_random_vector(int size) {
  std::uniform_real_distribution<double> unif(static_cast<double>(0), static_cast<double>(255));
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  std::vector<double> tmp;
  tmp.reserve(size);
  std::generate_n(std::back_inserter(tmp), size, [&]() { return unif(rand_engine); });

  return tmp;
}
}  // namespace veliev_e_sobel_operator_mpi_p

TEST(veliev_e_sobel_operator_perf_test, Testpipe) {
  boost::mpi::communicator world;
  int h = 25000;
  int w = 500;
  std::vector<double> in = veliev_e_sobel_operator_mpi_p::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<veliev_e_sobel_operator_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(veliev_e_sobel_operator_perf_test, Testtask) {
  boost::mpi::communicator world;
  int h = 25000;
  int w = 500;
  std::vector<double> in = veliev_e_sobel_operator_mpi_p::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<veliev_e_sobel_operator_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
