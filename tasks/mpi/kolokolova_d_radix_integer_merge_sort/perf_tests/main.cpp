// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kolokolova_d_radix_integer_merge_sort/include/ops_mpi.hpp"

namespace kolokolova_d_radix_integer_merge_sort_mpi {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(-10000, 10000);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace kolokolova_d_radix_integer_merge_sort_mpi

TEST(kolokolova_d_radix_integer_merge_sort_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size_vector = 120000;
  std::vector<int> unsorted_vector(size_vector);
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result(size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    unsorted_vector = kolokolova_d_radix_integer_merge_sort_mpi::getRandomVector(size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataPar->inputs_count.emplace_back(unsorted_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
    taskDataPar->outputs_count.emplace_back(sorted_vector.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(size_vector, int(sorted_vector.size()));
  }
}

TEST(kolokolova_d_radix_integer_merge_sort_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size_vector = 120000;
  std::vector<int> unsorted_vector(size_vector);
  std::vector<int32_t> sorted_vector(int(unsorted_vector.size()), 0);
  std::vector<int32_t> result(size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    unsorted_vector = kolokolova_d_radix_integer_merge_sort_mpi::getRandomVector(size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
    taskDataPar->inputs_count.emplace_back(unsorted_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
    taskDataPar->outputs_count.emplace_back(sorted_vector.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<kolokolova_d_radix_integer_merge_sort_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(size_vector, int(sorted_vector.size()));
  }
}