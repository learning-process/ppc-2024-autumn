// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <functional>
#include <numbers>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sorokin_a_graham_algorithm/include/ops_mpi.hpp"

namespace sorokin_a_graham_algorithm_mpi1 {
std::vector<int> getrndvec(int n, int radius) {
  if (n % 2 != 0) {
    throw std::invalid_argument("The number of elements n must be even.");
  }
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  std::uniform_real_distribution<double> dist_radius(0.0, static_cast<double>(radius));
  std::uniform_real_distribution<double> dist_angle(0.0, 2.0 * std::numbers::pi);
  std::vector<int> tmp(n);
  for (int i = 0; i < n / 2; ++i) {
    double r = dist_radius(rand_engine);
    double theta = dist_angle(rand_engine);
    double x = r * cos(theta);
    double y = r * sin(theta);
    tmp[2 * i] = static_cast<int>(x);
    tmp[2 * i + 1] = static_cast<int>(y);
  }

  return tmp;
}
}  // namespace sorokin_a_graham_algorithm_mpi1

TEST(sorokin_a_graham_algorithm_perf_test, test_10000000_points_tack) {
  boost::mpi::communicator world;
  std::vector<int> in = sorokin_a_graham_algorithm_mpi1::getrndvec(20000000, 200);
  std::vector<int> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<sorokin_a_graham_algorithm_mpi::TestMPITaskParallel>(taskDataPar);
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
TEST(sorokin_a_graham_algorithm_perf_test, test_10000000_points) {
  boost::mpi::communicator world;
  std::vector<int> in = sorokin_a_graham_algorithm_mpi1::getrndvec(20000000, 200);
  std::vector<int> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<sorokin_a_graham_algorithm_mpi::TestMPITaskParallel>(taskDataPar);
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
