#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kurakin_m_producer_consumer/include/kurakin_producer_consumer_ops_mpi.hpp"

TEST(kurakin_m_producer_consumer_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<kurakin_m_producer_consumer_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(data_counts.size(), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<kurakin_m_producer_consumer_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(data_counts.size(), data_counts_sum[0]);
  }
}
