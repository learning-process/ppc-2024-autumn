#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vedernikova_k_star_topology_mpi/include/ops_mpi.hpp"

namespace vedernikova_k_star_topology_mpi {
std::vector<int> make_random_vector(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (size_t i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
}  // namespace vedernikova_k_star_topology_mpi
TEST(vedernikova_k_star_topology_mpi_perf_test, vedernikova_k_test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    GTEST_SKIP();
    return;
  }
  // Create TaskData
  const size_t DataLength = 4096;

  std::vector<int> destinations(world.size() - 1);
  std::vector<int> data((world.size() - 1) * DataLength);

  std::vector<int> res(DataLength);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;

    destinations.resize(world.size() - 1);
    std::iota(destinations.begin(), destinations.end(), 1);
    std::shuffle(destinations.begin(), destinations.end(), rd);

    data = vedernikova_k_star_topology_mpi::make_random_vector((world.size() - 1) * DataLength);
  }
  boost::mpi::broadcast(world, destinations.data(), destinations.size(), 0);
  boost::mpi::broadcast(world, data.data(), data.size(), 0);
  if (world.rank() != 0) {
    auto &dst = destinations[world.rank() - 1];
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dst));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&(data[(dst - 1) * DataLength])));

    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(DataLength);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel = std::make_shared<vedernikova_k_star_topology_mpi::TestMPITaskParallel>(taskDataPar);
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
  }
}

TEST(vedernikova_k_star_topology_mpi_perf_test, vedernikova_k_test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    GTEST_SKIP();
    return;
  }
  // Create TaskData
  const size_t DataLength = 4096;

  std::vector<int> destinations(world.size() - 1);
  std::vector<int> data((world.size() - 1) * DataLength);

  std::vector<int> res(DataLength);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;

    destinations.resize(world.size() - 1);
    std::iota(destinations.begin(), destinations.end(), 1);
    std::shuffle(destinations.begin(), destinations.end(), rd);

    data = vedernikova_k_star_topology_mpi::make_random_vector((world.size() - 1) * DataLength);
  }
  boost::mpi::broadcast(world, destinations.data(), destinations.size(), 0);
  boost::mpi::broadcast(world, data.data(), data.size(), 0);
  if (world.rank() != 0) {
    auto &dst = destinations[world.rank() - 1];
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dst));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&(data[(dst - 1) * DataLength])));

    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(DataLength);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel = std::make_shared<vedernikova_k_star_topology_mpi::TestMPITaskParallel>(taskDataPar);
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
  }
}
