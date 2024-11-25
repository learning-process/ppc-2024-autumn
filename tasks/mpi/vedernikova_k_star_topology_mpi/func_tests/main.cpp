#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

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

TEST(vedernikova_k_star_topology_mpi, data_length_64) {
  const size_t DataLength = 64;
  boost::mpi::communicator world;
  if (world.size() < 3) {
    GTEST_SKIP();
    return;
  }

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

  vedernikova_k_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  int is_failed = 0;
  if (world.rank() != 0) {
    is_failed = std::equal(res.begin(), res.end(), data.begin() + (world.rank() - 1) * DataLength,
                           data.begin() + world.rank() * DataLength)
                    ? 0
                    : 1;
  }
  int failures = 0;
  boost::mpi::reduce(world, is_failed, failures, std::plus(), 0);
  EXPECT_EQ(failures, 0);
}
TEST(vedernikova_k_star_topology_mpi, data_length_0) {
  const size_t DataLength = 0;
  boost::mpi::communicator world;
  if (world.size() < 3) {
    GTEST_SKIP();
    return;
  }
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

  vedernikova_k_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  int is_failed = 0;
  if (world.rank() != 0) {
    is_failed = std::equal(res.begin(), res.end(), data.begin() + (world.rank() - 1) * DataLength,
                           data.begin() + world.rank() * DataLength)
                    ? 0
                    : 1;
  }
  int failures = 0;
  boost::mpi::reduce(world, is_failed, failures, std::plus(), 0);
  EXPECT_EQ(failures, 0);
}