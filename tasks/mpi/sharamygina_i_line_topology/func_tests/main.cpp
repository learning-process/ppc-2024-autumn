#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sharamygina_i_line_topology/include/ops_mpi.h"

namespace sharamygina_i_line_topology_mpi {
void generator(std::vector<int>& v) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dist(gen);
  }
}
}  // namespace sharamygina_i_line_topology_mpi

TEST(sharamygina_i_line_topology_mpi, checkTransferedData) {
  boost::mpi::communicator world;
  int size = 20000;
  auto sendler = 0;
  auto recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data(size);
  std::vector<int> received_data;

  if (world.rank() == sendler) {
    sharamygina_i_line_topology_mpi::generator(data);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (sendler != recipient) {
      world.send(recipient, 0, data);
    }
  }
  if (world.rank() == recipient) {
    if (sendler != recipient) {
      world.recv(sendler, 0, data);
    }

    received_data.resize(size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
  }

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);

  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == recipient) {
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
  }
}

TEST(sharamygina_i_line_topology_mpi, transferRandomData) {
  boost::mpi::communicator world;

  int size = 20000;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  int sendler = std::rand() % (world.size());
  int recipient = sendler + std::rand() % (world.size() - sendler);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data(size);
  std::vector<int> received_data;

  if (world.rank() == sendler) {
    sharamygina_i_line_topology_mpi::generator(data);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (sendler != recipient) {
      world.send(recipient, 0, data);
    }
  }
  if (world.rank() == recipient) {
    if (sendler != recipient) {
      world.recv(sendler, 0, data);
    }

    received_data.resize(size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
  }

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);

  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == recipient) {
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
  }
}

TEST(sharamygina_i_line_topology_mpi, insufficientInputs) {
  boost::mpi::communicator world;

  int size = 20000;
  int sendler = 0;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(size);

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_FALSE(testTask.validation());
}

TEST(sharamygina_i_line_topology_mpi, invalidSendler) {
  boost::mpi::communicator world;

  int size = 20000;
  int sendler = -1;
  int recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_FALSE(testTask.validation());
}

TEST(sharamygina_i_line_topology_mpi, invalidRecipient) {
  boost::mpi::communicator world;

  int size = 10000;
  int sendler = 0;
  int recipient = -1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_FALSE(testTask.validation());
}

TEST(sharamygina_i_line_topology_mpi, invalidNumberOfElements) {
  boost::mpi::communicator world;

  int size = -10;
  int sendler = 0;
  int recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_FALSE(testTask.validation());
}

TEST(sharamygina_i_line_topology_mpi, absenceOfInputData) {
  boost::mpi::communicator world;

  int size = 20000;
  int sendler = 0;
  int recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  if (world.rank() == sendler) {
    ASSERT_FALSE(testTask.validation());
  } else {
    SUCCEED();
  }
}

TEST(sharamygina_i_line_topology_mpi, absenceOutputDataOnRecipient) {
  boost::mpi::communicator world;

  int size = 20000;
  auto sendler = 0;
  auto recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data;
  if (world.rank() == sendler) {
    sharamygina_i_line_topology_mpi::generator(data);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  }

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  if (world.rank() == recipient) {
    ASSERT_FALSE(testTask.validation());
  } else {
    SUCCEED();
  }
}

TEST(sharamygina_i_line_topology_mpi, zeroNumberOfElements) {
  boost::mpi::communicator world;

  int size = 0;
  int sendler = 0;
  int recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_FALSE(testTask.validation());
}

TEST(sharamygina_i_line_topology_mpi, equalSenderAndRecipient) {
  boost::mpi::communicator world;

  int size = 20000;
  int sendler = 0;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data(size);
  std::vector<int> received_data;

  if (world.rank() == sendler) {
    sharamygina_i_line_topology_mpi::generator(data);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  }

  if (world.rank() == sendler) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
  }

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testTask.validation());
  }
}

TEST(sharamygina_i_line_topology_mpi, vectorOf1024) {
  boost::mpi::communicator world;

  int size = 1024;
  auto sendler = 0;
  auto recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data(size);
  std::vector<int> received_data;

  if (world.rank() == sendler) {
    sharamygina_i_line_topology_mpi::generator(data);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (sendler != recipient) {
      world.send(recipient, 0, data);
    }
  }
  if (world.rank() == recipient) {
    if (sendler != recipient) {
      world.recv(sendler, 0, data);
    }

    received_data.resize(size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
  }

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);

  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == recipient) {
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
  }
}

TEST(sharamygina_i_line_topology_mpi, smallSetVector) {
  boost::mpi::communicator world;

  int size = 12;
  auto sendler = 0;
  auto recipient = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(sendler);
  taskData->inputs_count.emplace_back(recipient);
  taskData->inputs_count.emplace_back(size);

  std::vector<int> data{1, 2, 4, -8, 16, 32, -64, 128, 256, -512, 1024, 2048};
  std::vector<int> received_data;

  if (world.rank() == sendler) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    if (sendler != recipient) {
      world.send(recipient, 0, data);
    }
  }
  if (world.rank() == recipient) {
    if (sendler != recipient) {
      world.recv(sendler, 0, data);
    }

    received_data.resize(size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
  }

  sharamygina_i_line_topology_mpi::line_topology_mpi testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);

  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == recipient) {
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
  }
}
