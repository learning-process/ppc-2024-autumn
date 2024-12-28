// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/ermilova_d_Shell_sort_simple_merge/include/ops_mpi.hpp"

static std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;

  const int size = 1000;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> sort_ref = input;
  std::sort(sort_ref.begin(), sort_ref.end());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(output, sort_ref);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;

  const int size = 1000;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> sort_ref = input;
  std::sort(sort_ref.begin(), sort_ref.end());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(output, sort_ref);
  }
}
