#include <gtest/gtest.h>

#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"
#include "mpi/agafeev_s_linear_topology/include/lintop_mpi.hpp"

template <typename T>
static std::vector<T> create_RandomVector(int size) {
  auto rand_gen = std::mt19937(std::time(nullptr));
  std::vector<T> vec(size);
  for (unsigned int i = 0; i < vec.size(); ++i) vec[i] = rand_gen() % 200 - 100;

  return vec;
}

TEST(agafeev_s_linear_topology, test_empty_matrix) {
  boost::mpi::communicator world;
  std::vector<int> in_matrix(0);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<int>(0);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<int>>(taskData);

  if (world.rank() == 0) {
    ASSERT_EQ(testTask->validation(), false);
  }
}

TEST(agafeev_s_linear_topology, test_find_in_1x1_matrix) {
  boost::mpi::communicator world;
  std::vector<int> in_matrix(9);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<int>(1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(out[0]);
  }
}

TEST(agafeev_s_linear_topology, test_find_in_3x3_matrix) {
  boost::mpi::communicator world;
  std::vector<int> in_matrix(9);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<int>(9);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(out[0]);
  }
}

TEST(agafeev_s_linear_topology, test_find_in_100x100_matrix) {
  boost::mpi::communicator world;

  std::vector<int> in_matrix(10000);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<int>(10000);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(out[0]);
  }
}

TEST(agafeev_s_linear_topology, test_find_in_1000x12_matrix) {
  boost::mpi::communicator world;

  std::vector<int> in_matrix(12000);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<int>(12000);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(out[0]);
  }
}

TEST(agafeev_s_linear_topology, test_find_in_100x100_matrix_double) {
  boost::mpi::communicator world;

  std::vector<double> in_matrix(100 * 100);
  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<double>(10000);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataMpi->inputs_count.emplace_back(in_matrix.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMpi->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<double>>(taskDataMpi);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(out[0]);
  }
}

TEST(agafeev_s_linear_topology, test_find_in_9x45_matrix_double) {
  boost::mpi::communicator world;

  std::vector<double> in_matrix(9 * 45);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<double>(9 * 45);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataMpi->inputs_count.emplace_back(in_matrix.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMpi->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<double>>(taskDataMpi);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(out[0]);
  }
}

TEST(agafeev_s_linear_topology, test_find_in_300x200_matrix_double) {
  boost::mpi::communicator world;

  std::vector<double> in_matrix(9 * 45);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomVector<double>(9 * 45);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataMpi->inputs_count.emplace_back(in_matrix.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMpi->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology<double>>(taskDataMpi);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(out[0]);
  }
}