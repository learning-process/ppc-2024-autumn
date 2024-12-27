// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kovalchuk_a_max_of_vector_elements/include/ops_mpi.hpp"

using namespace kovalchuk_a_max_of_vector_elements;

static std::vector<int> getRandomVector(int sz, int min = MINIMALGEN, int max = MAXIMUMGEN);
static std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int min = MINIMALGEN, int max = MAXIMUMGEN);

static std::vector<int> getRandomVector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

static std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int min, int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVector(columns, min, max);
  }
  return vec;
}

TEST(kovalchuk_a_max_of_vector_elements, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  int ref = INT_MAX;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    int count_rows = 1000;
    int count_columns = 1000;
    global_matrix = getRandomMatrix(count_rows, count_columns);
    size_t index = gen() % (static_cast<size_t>(count_rows) * count_columns);
    global_matrix[index / count_columns][index % count_columns] = ref;
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  auto testMpiTaskParallel = std::make_shared<kovalchuk_a_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(ref, global_max[0]);
  }
}

TEST(kovalchuk_a_max_of_vector_elements, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  int ref = INT_MAX;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    int count_rows = 3;
    int count_columns = 3;
    global_matrix = getRandomMatrix(count_rows, count_columns);
    int index = gen() % (count_rows * count_columns);
    global_matrix[index / count_columns][index / count_rows] = ref;
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  auto testMpiTaskParallel = std::make_shared<kovalchuk_a_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(ref, global_max[0]);
  }
}
