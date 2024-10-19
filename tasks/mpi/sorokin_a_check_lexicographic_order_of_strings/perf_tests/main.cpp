// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sorokin_a_check_lexicographic_order_of_strings/include/ops_mpi.hpp"

TEST(sorokin_a_check_lexicographic_order_of_strings_mpi, test_basic) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> strs = {{'a', 'p', 'p', 'p'}, {'b', 'a', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataPar->inputs_count.emplace_back(strs.size());
    taskDataPar->inputs_count.emplace_back(strs[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(res[0], 0);
  }
}

TEST(sorokin_a_check_lexicographic_order_of_strings_mpi, test_basic2) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> strs = {{'a', 'a', 'z', 'p'}, {'a', 'a', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataPar->inputs_count.emplace_back(strs.size());
    taskDataPar->inputs_count.emplace_back(strs[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(res[0], 1);
  }
}
