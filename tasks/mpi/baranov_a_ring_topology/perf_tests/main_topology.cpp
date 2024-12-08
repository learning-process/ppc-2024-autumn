#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/baranov_a_ring_topology/src/source_topology.cpp"

TEST(mpi_baranov_a_ring_topology_perf_test, test_pipeline_run) {
  const int count_size_vector = 10000000;
  boost::mpi::communicator world;
  std::vector<int> arr(count_size_vector);
  std::vector<int> out_vec(count_size_vector);
  std::vector<int> out_poll;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    out_poll = std::vector<int>(world.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    taskDataPar->outputs_count.emplace_back(2);
  }
  auto testMpiTaskParallel = std::make_shared<baranov_a_ring_topology_mpi::ring_topology<int>>(taskDataPar);
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
    ASSERT_EQ(arr, out_vec);
  }
}
TEST(mpi_baranov_a_ring_topology_perf_test, test_task_run) {
  const int count_size_vector = 10000000;
  boost::mpi::communicator world;
  std::vector<int> arr(count_size_vector);
  std::vector<int> out_vec(count_size_vector);
  std::vector<int> out_poll;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    out_poll = std::vector<int>(world.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    taskDataPar->outputs_count.emplace_back(2);
  }
  auto testMpiTaskParallel = std::make_shared<baranov_a_ring_topology_mpi::ring_topology<int>>(taskDataPar);
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
    ASSERT_EQ(arr, out_vec);
  }
}
