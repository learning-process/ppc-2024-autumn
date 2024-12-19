#include <gtest/gtest.h>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"
#include "mpi/agafeev_s_linear_topology/include/lintop_mpi.hpp"

TEST(agafeev_s_linear_topology, check_wrong_input) {
  boost::mpi::communicator world;
  int sender = -1;
  int receiver = world.size() - 1;
  std::vector<int> right_route = agafeev_s_linear_topology::calculating_Route(sender, receiver);
  bool out = false;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sender));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&receiver));

  if (world.rank() == sender) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_route.data()));
    taskData->inputs_count.emplace_back(right_route.size());

  } else {
    if (world.rank() == receiver) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
      taskData->outputs_count.emplace_back(1);
    }
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology>(taskData);
  ASSERT_FALSE(testTask->validation());
}

TEST(agafeev_s_linear_topology, test_0_to_N) {
  boost::mpi::communicator world;
  int sender = 0;
  int receiver = world.size() - 1;
  if (world.size() <= std::max(sender, receiver)) {
    GTEST_SKIP();
  }

  std::vector<int> right_route = agafeev_s_linear_topology::calculating_Route(sender, receiver);
  bool out = false;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sender));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&receiver));

  if (world.rank() == sender) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_route.data()));
    taskData->inputs_count.emplace_back(right_route.size());
  }
  if (world.rank() == receiver) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskData->outputs_count.emplace_back(1);
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology>(taskData);
  ASSERT_TRUE(testTask->validation());
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == receiver) {
    ASSERT_TRUE(out);
  }
}

TEST(agafeev_s_linear_topology, test_2_to_1) {
  boost::mpi::communicator world;
  int sender = 2;
  int receiver = 1;
  if (world.size() <= std::max(sender, receiver)) {
    GTEST_SKIP();
  }

  std::vector<int> right_route = agafeev_s_linear_topology::calculating_Route(sender, receiver);
  bool out = false;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sender));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&receiver));

  if (world.rank() == sender) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_route.data()));
    taskData->inputs_count.emplace_back(right_route.size());
  } else {
    if (world.rank() == receiver) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
      taskData->outputs_count.emplace_back(1);
    }
  }

  auto testTask = std::make_shared<agafeev_s_linear_topology::LinearTopology>(taskData);
  ASSERT_TRUE(testTask->validation());
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == receiver) {
    ASSERT_TRUE(out);
  }
}

/*
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
}*/
/*
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
  std::cout << "After run: " << out[0] << std::endl;
  testTask->post_processing();
  std::cout << "Post proc " << out[0] << std::endl;
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
}*/