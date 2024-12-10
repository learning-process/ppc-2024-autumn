// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/rysev_m_hypercube/include/ops_mpi.hpp"

TEST(rysev_m_gypercube, data_transfer_0_to_1) {
  boost::mpi::communicator world;

  if (world.size() < 2 || (world.size() & (world.size() - 1)) != 0) {
    GTEST_SKIP();
  }

  int _data = 10;
  int _sender = 0;
  int _target = 1;
  int out = -1;
  std::vector<int> out_path(log2(world.size()) + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_target));
  taskDataPar->inputs_count.emplace_back(1);
  if (world.rank() == _sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_data));
    taskDataPar->inputs_count.emplace_back(1);
  }
  if (world.rank() == _target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_path.data()));
    taskDataPar->outputs_count.emplace_back(out_path.size());
  }
  rysev_m_gypercube::GyperCube task(taskDataPar);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == _target) {
    out_path.erase(std::remove(out_path.begin(), out_path.end(), -1), out_path.end());
    world.send(_sender, 0, out);
    world.send(_sender, 0, out_path);
  }
  if (world.rank() == _sender) {
    std::vector<int> exp_path{0, 1};
    world.recv(_target, 0, out);
    world.recv(_target, 0, out_path);
    ASSERT_EQ(_data, out);
    ASSERT_EQ(exp_path, out_path);
  }
  std::cout << "TEST #1 " << world.rank() << "/" << world.size() - 1 << std::endl;
}

TEST(rysev_m_gypercube, data_transfer_1_to_3) {
  boost::mpi::communicator world;

  if ((world.size() & (world.size() - 1)) != 0 || log2(world.size()) < 2) {
    GTEST_SKIP();
  }

  int _data = 10;
  int _sender = 1;
  int _target = 3;
  int out = -1;
  std::vector<int> out_path(log2(world.size()) + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_target));
  taskDataPar->inputs_count.emplace_back(1);
  if (world.rank() == _sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_data));
    taskDataPar->inputs_count.emplace_back(1);
  }
  if (world.rank() == _target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_path.data()));
    taskDataPar->outputs_count.emplace_back(out_path.size());
  }
  rysev_m_gypercube::GyperCube task(taskDataPar);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == _target) {
    out_path.erase(std::remove(out_path.begin(), out_path.end(), -1), out_path.end());
    world.send(_sender, 0, out);
    world.send(_sender, 0, out_path);
  }
  if (world.rank() == _sender) {
    std::vector<int> exp_path{1, 3};
    world.recv(_target, 0, out);
    world.recv(_target, 0, out_path);
    ASSERT_EQ(_data, out);
    ASSERT_EQ(exp_path, out_path);
  }
  std::cout << "TEST #2 " << world.rank() << "/" << world.size() - 1 << std::endl;
}

TEST(rysev_m_gypercube, data_transfer_3_to_0) {
  boost::mpi::communicator world;

  if ((world.size() & (world.size() - 1)) != 0 || log2(world.size()) < 2) {
    GTEST_SKIP();
  }

  int _data = 10;
  int _sender = 3;
  int _target = 0;
  int out = -1;
  std::vector<int> out_path(log2(world.size()) + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_target));
  taskDataPar->inputs_count.emplace_back(1);
  if (world.rank() == _sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_data));
    taskDataPar->inputs_count.emplace_back(1);
  }
  if (world.rank() == _target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_path.data()));
    taskDataPar->outputs_count.emplace_back(out_path.size());
  }
  rysev_m_gypercube::GyperCube task(taskDataPar);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == _target) {
    out_path.erase(remove(out_path.begin(), out_path.end(), -1), out_path.end());
    world.send(_sender, 0, out);
    world.send(_sender, 0, out_path);
  }
  if (world.rank() == _sender) {
    std::vector<int> exp_path{3, 2, 0};
    world.recv(_target, 0, out);
    world.recv(_target, 0, out_path);
    ASSERT_EQ(_data, out);
    ASSERT_EQ(exp_path, out_path);
  }
  std::cout << "TEST #3 " << world.rank() << "/" << world.size() - 1 << std::endl;
}

TEST(rysev_m_gypercube, data_transfer_0_to_3) {
  boost::mpi::communicator world;

  if ((world.size() & (world.size() - 1)) != 0 || log2(world.size()) < 2) {
    GTEST_SKIP();
  }

  int _data = 10;
  int _sender = 0;
  int _target = 3;
  int out = -1;
  std::vector<int> out_path(log2(world.size()) + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_target));
  taskDataPar->inputs_count.emplace_back(1);
  if (world.rank() == _sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_data));
    taskDataPar->inputs_count.emplace_back(1);
  }
  if (world.rank() == _target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_path.data()));
    taskDataPar->outputs_count.emplace_back(out_path.size());
  }
  rysev_m_gypercube::GyperCube task(taskDataPar);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == _target) {
    out_path.erase(std::remove(out_path.begin(), out_path.end(), -1), out_path.end());
    world.send(_sender, 0, out);
    world.send(_sender, 0, out_path);
  }
  if (world.rank() == _sender) {
    std::vector<int> exp_path{0, 1, 3};
    world.recv(_target, 0, out);
    world.recv(_target, 0, out_path);
    ASSERT_EQ(_data, out);
    ASSERT_EQ(exp_path, out_path);
  }
  std::cout << "TEST #4 " << world.rank() << "/" << world.size() - 1 << std::endl;
}

TEST(rysev_m_gypercube, data_transfer_0_to_7) {
  boost::mpi::communicator world;

  if ((world.size() & (world.size() - 1)) != 0 || log2(world.size()) < 3) {
    GTEST_SKIP();
  }

  int _data = 10;
  int _sender = 0;
  int _target = 7;
  int out = -1;
  std::vector<int> out_path(log2(world.size()) + 1, -1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_target));
  taskDataPar->inputs_count.emplace_back(1);
  if (world.rank() == _sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&_data));
    taskDataPar->inputs_count.emplace_back(1);
  }
  if (world.rank() == _target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_path.data()));
    taskDataPar->outputs_count.emplace_back(out_path.size());
  }
  rysev_m_gypercube::GyperCube task(taskDataPar);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == _target) {
    out_path.erase(std::remove(out_path.begin(), out_path.end(), -1), out_path.end());
    world.send(_sender, 0, out);
    world.send(_sender, 0, out_path);
  }
  if (world.rank() == _sender) {
    std::vector<int> exp_path{0, 1, 3, 7};
    world.recv(_target, 0, out);
    world.recv(_target, 0, out_path);
    ASSERT_EQ(_data, out);
    ASSERT_EQ(exp_path, out_path);
  }
  std::cout << "TEST #5 " << world.rank() << "/" << world.size() - 1 << std::endl;
}