// Copyright 2024 Kabalova Valeria
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/kabalova_v_my_reduce/include/kabalova_my_reduce.hpp"

namespace kabalova_v_my_reduce {
void createVector(std::vector<int>& vec) {
  vec[0] = 1;
  for (size_t i = 1; i < vec.size(); i++) {
    vec[i] = vec[i - 1] + i;
  }
}
}  // namespace kabalova_v_my_reduce

TEST(kabalova_v_my_reduce, test_pipeline_run) {
  boost::mpi::communicator world;
  size_t vecSize = 200000;
  std::vector<int> vec(vecSize);
  kabalova_v_my_reduce::createVector(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  auto testMpiTaskParallel = std::make_shared<kabalova_v_my_reduce::TestMPITaskParallel>(taskDataMpi, "+");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5000;
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

TEST(kabalova_v_my_reduce, test_task_run) {
  boost::mpi::communicator world;
  size_t vecSize = 200000;
  std::vector<int> vec(vecSize);
  kabalova_v_my_reduce::createVector(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  auto testMpiTaskParallel = std::make_shared<kabalova_v_my_reduce::TestMPITaskParallel>(taskDataMpi, "+");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5000;
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