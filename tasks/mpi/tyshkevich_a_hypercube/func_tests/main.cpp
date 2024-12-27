#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/tyshkevich_a_hypercube/include/ops_mpi.hpp"

namespace tyshkevich_a_hypercube_mpi {

void getRandomIntPair(int minValue, int maxValue, int& sender, int& target) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(minValue, maxValue);

  sender = dist(gen);
  do {
    target = dist(gen);
  } while (target == sender);
}

std::vector<int> getRandomIntVector(int size, int minValue = 0, int maxValue = 100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(minValue, maxValue);

  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

void run_test(boost::mpi::communicator& world, int sender, int target, int data_size = 10) {
  std::vector<int> input_vector(data_size);
  std::vector<int> output_result(data_size);

  int path_length;
  int dimension = static_cast<int>(std::log2(world.size()));
  std::vector<int> shortest_route;
  int current = sender;
  do {
    shortest_route.push_back(current);
    current = tyshkevich_a_hypercube_mpi::getNextNode(current, target, dimension);
  } while (current != target);
  shortest_route.push_back(target);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sender));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == sender) {
    input_vector = tyshkevich_a_hypercube_mpi::getRandomIntVector(data_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(data_size);
  } else if (world.rank() == target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
    taskDataPar->outputs_count.emplace_back(data_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&path_length));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto taskParallel = std::make_shared<tyshkevich_a_hypercube_mpi::HypercubeParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == target) {
    world.send(sender, 0, output_result);
    world.send(sender, 1, path_length);
  } else if (world.rank() == sender) {
    world.recv(target, 0, output_result);
    world.recv(target, 1, path_length);
    EXPECT_EQ(input_vector, output_result);
    EXPECT_EQ(static_cast<int>(shortest_route.size()), path_length);
  }
}

bool initial_test(int world_size, int sender, int target) {
  int init_dimension = static_cast<int>(std::log2(world_size));
  return (1 << init_dimension) == world_size && sender < world_size && target < world_size;
}

}  // namespace tyshkevich_a_hypercube_mpi

TEST(tyshkevich_a_hypercube_mpi, 0_to_1_process_send) {
  boost::mpi::communicator world;
  int sender = 0;
  int target = 1;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 1_to_0_process_send) {
  boost::mpi::communicator world;
  int sender = 1;
  int target = 0;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 0_to_2_process_send) {
  boost::mpi::communicator world;
  int sender = 0;
  int target = 2;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 2_to_0_process_send) {
  boost::mpi::communicator world;
  int sender = 2;
  int target = 0;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 0_to_3_process_send) {
  boost::mpi::communicator world;
  int sender = 0;
  int target = 3;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 3_to_0_process_send) {
  boost::mpi::communicator world;
  int sender = 3;
  int target = 0;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 1_to_3_process_send) {
  boost::mpi::communicator world;
  int sender = 1;
  int target = 3;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 3_to_1_process_send) {
  boost::mpi::communicator world;
  int sender = 3;
  int target = 1;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 2_to_3_process_send) {
  boost::mpi::communicator world;
  int sender = 2;
  int target = 3;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, 3_to_2_process_send) {
  boost::mpi::communicator world;
  int sender = 3;
  int target = 2;

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}

TEST(tyshkevich_a_hypercube_mpi, random_send_between_0_and_max_data) {
  boost::mpi::communicator world;
  int sender;
  int target;
  if (world.rank() == 0 && world.size() > 1) {
    tyshkevich_a_hypercube_mpi::getRandomIntPair(0, world.size() - 1, sender, target);
  }
  boost::mpi::broadcast(world, sender, 0);
  boost::mpi::broadcast(world, target, 0);

  if (tyshkevich_a_hypercube_mpi::initial_test(world.size(), sender, target)) {
    tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
  } else {
    GTEST_SKIP();
  }
}