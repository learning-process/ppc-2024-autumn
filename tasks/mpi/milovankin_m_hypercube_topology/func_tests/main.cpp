#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>
#include <vector>

#include "mpi/milovankin_m_hypercube_topology/include/ops_mpi.hpp"

namespace milovankin_m_hypercube_topology {
static void run_test_parallel(const std::string& data, int dest, std::vector<int> route_expected = {}) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;  // tests are designed for 4+ processes

  milovankin_m_hypercube_topology::Hypercube::DataIn data_in_struct(data, dest);
  milovankin_m_hypercube_topology::Hypercube::DataIn data_out_struct;

  if (route_expected.empty()) {
    route_expected = milovankin_m_hypercube_topology::Hypercube::calculate_route(dest);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
    taskDataPar->outputs_count.emplace_back(1);
  }

  milovankin_m_hypercube_topology::Hypercube Hypercube(taskDataPar);

  // Edge cases
  if ((world.size() & (world.size() - 1)) != 0) {  // not a power of 2
    ASSERT_FALSE(Hypercube.validation());
    return;
  }
  if (dest == 0 || dest >= world.size()) {
    ASSERT_TRUE(Hypercube.validation());
    if (world.rank() == 0) {
      ASSERT_FALSE(Hypercube.pre_processing());
    }
    return;
  }

  ASSERT_TRUE(Hypercube.validation());
  ASSERT_TRUE(Hypercube.pre_processing());

  Hypercube.run();
  Hypercube.post_processing();

  // Assert
  if (world.rank() == 0) {
    ASSERT_EQ(data_out_struct.data, data_in_struct.data);
    ASSERT_EQ(data_out_struct.route, route_expected);
  }
}
}  // namespace milovankin_m_hypercube_topology

TEST(milovankin_m_hypercube_topology, calculate_route_tests) {
  boost::mpi::communicator world;
  if (world.rank() != 0) GTEST_SKIP();

  std::vector<int> expect;

  expect = {0};  // 0
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(0), expect);

  expect = {0, 1};  // 0 -> 1
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(1), expect);

  expect = {0, 2};  // 0 -> 10
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(2), expect);

  expect = {0, 1, 3};  // 00 -> 01 -> 11
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(3), expect);

  expect = {0, 4};  // 00 -> 100
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(4), expect);

  expect = {0, 1, 5};  // 000 -> 001 -> 101
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(5), expect);

  expect = {0, 1, 3, 7};  // 000 -> 001 -> 011 -> 111
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(7), expect);

  expect = {0, 2, 6, 14};  // 0000 -> 0010 -> 0110 -> 1110
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(14), expect);

  expect = {0, 1, 5, 13, 29};  // 00000 -> 00001 -> 000101 -> 01101 -> 11101
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_route(29), expect);
}

TEST(milovankin_m_hypercube_topology, same_source_and_destination) {
  milovankin_m_hypercube_topology::run_test_parallel("something something", 0, {0});
}

TEST(milovankin_m_hypercube_topology, target_process_out_of_range) {
  milovankin_m_hypercube_topology::run_test_parallel("something something", 999999, {0});
}

TEST(milovankin_m_hypercube_topology, validation_failed_wrong_task_data) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  milovankin_m_hypercube_topology::Hypercube Hypercube(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(Hypercube.validation());
  }
}

TEST(milovankin_m_hypercube_topology, normal_input_1) {
  milovankin_m_hypercube_topology::run_test_parallel("aaabbbcccddd", 1, {0, 1});
}

TEST(milovankin_m_hypercube_topology, normal_input_2) {
  milovankin_m_hypercube_topology::run_test_parallel("Hiiii :33", 3, {0, 1, 3});
}

TEST(milovankin_m_hypercube_topology, normal_input_3) {
  milovankin_m_hypercube_topology::run_test_parallel("ABCDE", 2, {0, 2});
}

TEST(milovankin_m_hypercube_topology, large_string) {
  std::string large_str(1'000'000, 'a');
  milovankin_m_hypercube_topology::run_test_parallel(large_str, 3, {0, 1, 3});
}

TEST(milovankin_m_hypercube_topology, any_processor_count_auto_test) {
  boost::mpi::communicator world;
  int dest = world.size() / 3 * 2;
  std::vector<int> expected_route = milovankin_m_hypercube_topology::Hypercube::calculate_route(dest);
  milovankin_m_hypercube_topology::run_test_parallel("123 456 789", dest, expected_route);
}
