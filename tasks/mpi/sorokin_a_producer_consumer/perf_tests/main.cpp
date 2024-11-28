#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sorokin_a_producer_consumer/include/ops_mpi.hpp"

TEST(sorokin_a_producer_consumer_mpi, Test_Basic) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < (world.size() - 1) / 2; i++) global_vec.push_back(i + 1);
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<sorokin_a_producer_consumer_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } else {
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
    for (int i = 0; i < (world.size() - 1) / 2; i++) ASSERT_EQ(global_vec[i], global_sum[i]);
  }
}
