#include <boost/mpi/timer.hpp>

#include "../include/tests.hpp"
#include "core/perf/include/perf.hpp"

TEST(khasanyanov_k_ring_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    SUCCEED();
    return;
  }
  // prime size
  const int size = 2039;
  const std::vector<int> in_data = khasanyanov_k_ring_topology_mpi::generate_random_vector<int>(size, -500, 500);
  std::vector<int> out_data(in_data);
  std::vector<int> order(world.size());

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_ring_topology_mpi::create_task_data<int>(out_data, order);
  }

  auto testTask = std::make_shared<khasanyanov_k_ring_topology_mpi::RingTopology<int>>(taskData);

  RUN_TASK(*testTask);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto pattern_order = khasanyanov_k_ring_topology_mpi::RingTopology<int>::true_order(world.size());
    ASSERT_EQ(pattern_order, order);
    ASSERT_EQ(in_data, out_data);
  }
}

TEST(khasanyanov_k_ring_topology_mpi, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    SUCCEED();
    return;
  }
  // size 2^n
  const int size = 2048;
  const std::vector<int> in_data = khasanyanov_k_ring_topology_mpi::generate_random_vector<int>(size, -500, 500);
  std::vector<int> out_data(in_data);
  std::vector<int> order(world.size());

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_ring_topology_mpi::create_task_data<int>(out_data, order);
  }

  auto testTask = std::make_shared<khasanyanov_k_ring_topology_mpi::RingTopology<int>>(taskData);

  RUN_TASK(*testTask);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto pattern_order = khasanyanov_k_ring_topology_mpi::RingTopology<int>::true_order(world.size());
    ASSERT_EQ(pattern_order, order);
    ASSERT_EQ(in_data, out_data);
  }
}