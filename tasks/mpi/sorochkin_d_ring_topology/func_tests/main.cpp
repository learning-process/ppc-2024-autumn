#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

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

TEST(sorochkin_d_ring_topology_mpi_test, test_size_120) {
  boost::mpi::communicator world;
  std::vector<int> input;
  std::vector<int> output;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    input = sorochkin_d_ring_topology_mpi_test::getRandomVector(count_size_vector);
    output.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  sorochkin_d_test_task_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(input, output);
  }
}

TEST(sorochkin_d_ring_topology_mpi_test, test_size_200) {
  boost::mpi::communicator world;
  std::vector<int> input;
  std::vector<int> output;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    input = sorochkin_d_ring_topology_mpi_test::getRandomVector(count_size_vector);
    output.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  sorochkin_d_test_task_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(input, output);
  }
}

TEST(sorochkin_d_ring_topology_mpi_test, test_size_400) {
  boost::mpi::communicator world;
  std::vector<int> input;
  std::vector<int> output;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    input = sorochkin_d_ring_topology_mpi_test::getRandomVector(count_size_vector);
    output.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  sorochkin_d_test_task_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(input, output);
  }
}

TEST(sorochkin_d_ring_topology_mpi_test, test_size_0) {
  boost::mpi::communicator world;
  std::vector<int> input;
  std::vector<int> output;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    input = sorochkin_d_ring_topology_mpi_test::getRandomVector(count_size_vector);
    output.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  sorochkin_d_test_task_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(input, output);
  }
}