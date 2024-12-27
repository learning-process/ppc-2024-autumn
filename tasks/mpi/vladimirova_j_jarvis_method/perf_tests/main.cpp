// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vladimirova_j_jarvis_method/include/ops_mpi.hpp"

namespace mpi_vladimirova_j_jarvis_method_mpi {
std::vector<int> getRandomVal(size_t col, size_t row, size_t n) {
  std::vector<int> ans(row * col, 255);

  std::random_device dev;
  std::mt19937 gen(dev());
  for (size_t i = 0; i < n; i++) {
    int c = gen() % col;
    int r = gen() % row;
    ans[r * row + c] = 0;
  }
  return ans;
}

}  // namespace mpi_vladimirova_j_jarvis_method_mpi

TEST(mpi_vladimirova_j_jarvis_method_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  const int count = 6000;
  const int sz = 15000;
  std::vector<int32_t> global_ans(count, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = mpi_vladimirova_j_jarvis_method_mpi::getRandomVal(sz, sz, count / 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<vladimirova_j_jarvis_method_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
  }

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
    bool ans = (sz >= taskDataPar->outputs_count[0]);
    ASSERT_EQ(ans, true);
  }
}

TEST(mpi_vladimirova_j_jarvis_method_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  const int count = 6000;
  const int sz = 15000;
  std::vector<int> global_vec;
  std::vector<int32_t> global_ans(count, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = mpi_vladimirova_j_jarvis_method_mpi::getRandomVal(sz, sz, count / 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<vladimirova_j_jarvis_method_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
  }
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
    ASSERT_EQ(count >= taskDataPar->outputs_count[0], true);
  }
}
