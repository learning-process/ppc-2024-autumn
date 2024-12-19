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

TEST(agafeev_s_linear_topology, test_0_to_N_minus_1) {
  boost::mpi::communicator world;
  int sender = 0;
  int receiver = world.size() - 1;

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

  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}

TEST(agafeev_s_linear_topology, test_N_minus_1_to_0) {
  boost::mpi::communicator world;
  int sender = world.size() - 1;
  int receiver = 0;

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

  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}

TEST(agafeev_s_linear_topology, test_2_to_1) {
  boost::mpi::communicator world;
  int sender = 2;
  int receiver = 1;

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
  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}

TEST(agafeev_s_linear_topology, test_1_to_2) {
  boost::mpi::communicator world;
  int sender = 1;
  int receiver = 2;

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
  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}

TEST(agafeev_s_linear_topology, test_0_to_2) {
  boost::mpi::communicator world;
  int sender = 0;
  int receiver = 2;

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
  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}

TEST(agafeev_s_linear_topology, test_1_to_3) {
  boost::mpi::communicator world;
  int sender = 1;
  int receiver = 3;

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
  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}

TEST(agafeev_s_linear_topology, test_minus10_to_30) {
  boost::mpi::communicator world;
  int sender = -10;
  int receiver = 30;

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
  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}

TEST(agafeev_s_linear_topology, test_130_to_330) {
  boost::mpi::communicator world;
  int sender = 130;
  int receiver = 330;

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
  if (world.size() <= std::max(sender, receiver))
    ASSERT_FALSE(testTask->validation());
  else {
    ASSERT_TRUE(testTask->validation());
    testTask->pre_processing();
    testTask->run();
    testTask->post_processing();

    if (world.rank() == receiver) {
      ASSERT_TRUE(out);
    }
  }
}