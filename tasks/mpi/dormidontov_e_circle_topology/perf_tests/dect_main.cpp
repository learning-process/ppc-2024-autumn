#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/dormidontov_e_circle_topology/src/dect_ops_mpi.cpp"

TEST(dormidontov_e_circle_topology_mpi, test_pipeline_run) {
  const int size = 10000000;
  std::vector<int> mas(size, 1);
  std::vector<int> ans(size);
  std::vector<int> marks;
  boost::mpi::communicator world;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    marks.resize(world.rank() + 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  auto testMpiTaskParallel = std::make_shared<dormidontov_e_circle_topology_mpi::topology>(taskDataPar);
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
    ASSERT_EQ(mas, ans);
  }
}
TEST(dormidontov_e_circle_topology_mpi, test_task_run) {
  const int size = 10000000;
  std::vector<int> mas(size, 1);
  std::vector<int> ans(size);
  std::vector<int> marks;
  boost::mpi::communicator world;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    marks.resize(world.rank() + 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  auto testMpiTaskParallel = std::make_shared<dormidontov_e_circle_topology_mpi::topology>(taskDataPar);
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
    ASSERT_EQ(mas, ans);
  }
}