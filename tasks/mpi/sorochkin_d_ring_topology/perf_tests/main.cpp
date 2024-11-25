#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sorochkin_d_ring_topology/include/ops_mpi.hpp"

namespace sorochkin_d_ring_topology_mpi_test {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
}  // namespace sorochkin_d_ring_topology_mpi_test

TEST(sorochkin_d_ring_topology_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> input;
  std::vector<int> output;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 1024;
    input = sorochkin_d_ring_topology_mpi_test::getRandomVector(count_size_vector);
    output.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  auto testMpiTaskParallel = std::make_shared<sorochkin_d_test_task_mpi::TestMPITaskParallel<int>>(taskDataPar);
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
    ASSERT_EQ(input, output);
  }
}

TEST(sorochkin_d_ring_topology_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> input;
  std::vector<int> output;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 1024;
    input = sorochkin_d_ring_topology_mpi_test::getRandomVector(count_size_vector);
    output.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  auto testMpiTaskParallel = std::make_shared<sorochkin_d_test_task_mpi::TestMPITaskParallel<int>>(taskDataPar);
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
    ASSERT_EQ(input, output);
  }
}
