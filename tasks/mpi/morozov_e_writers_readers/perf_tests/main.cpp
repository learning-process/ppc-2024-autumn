#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/morozov_e_writers_readers/include/ops_mpi.hpp"

TEST(morozov_e_writers_readers, test_pipeline_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  int countIteration = INT_MAX;
  std::vector<int> ans{1};
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{countIteration}));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<morozov_e_writers_readers::TestMPITaskParallel>(data);
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
    ASSERT_EQ(ans[0], 0);
  }
}
TEST(morozov_e_writers_readers, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> vec{0};
  int countIteration = INT_MAX;
  std::vector<int> ans{1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{countIteration}));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<morozov_e_writers_readers::TestMPITaskParallel>(data);
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
    ASSERT_EQ(ans[0], 0);
  }
}
