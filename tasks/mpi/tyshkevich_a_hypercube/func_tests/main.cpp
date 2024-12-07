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

void run_test(boost::mpi::communicator& world, int sender, int target, int data_size = 100) {
  int n = static_cast<int>(std::log2(world.size()));
  std::vector<int> shortest_route;
  int current = sender;
  do {
    shortest_route.push_back(current);
    current = tyshkevich_a_hypercube_mpi::getNextNode(current, target, n);
  } while (current != target);

  shortest_route.push_back(target);

  std::vector<int> input_vector;
  std::vector<int> output_result(data_size);
  std::vector<int> output_check(data_size);
  std::vector<int> output_route(shortest_route.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sender));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == sender) {
    input_vector = tyshkevich_a_hypercube_mpi::getRandomIntVector(data_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
  } else if (world.rank() == target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
    taskDataPar->outputs_count.emplace_back(output_result.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_check.data()));
    taskDataPar->outputs_count.emplace_back(output_check.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_route.data()));
    taskDataPar->outputs_count.emplace_back(output_route.size());
  }

  auto taskParallel = std::make_shared<tyshkevich_a_hypercube_mpi::HypercubeParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == target) {
    EXPECT_EQ(output_check, output_result);
    EXPECT_EQ(shortest_route, output_route);
  }
}

bool initial_test(int world_size) {
  int init_dimension = static_cast<int>(std::log2(world_size));
  return (1 << init_dimension) == world_size;
}

}  // namespace tyshkevich_a_hypercube_mpi

TEST(tyshkevich_a_hypercube_mpi, 0_to_1_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 0;
    int target = 1;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 1_to_0_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 1;
    int target = 0;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 0_to_2_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 0;
    int target = 2;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 2_to_0_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 2;
    int target = 0;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 0_to_3_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 0;
    int target = 3;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 3_to_0_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 3;
    int target = 0;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 1_to_3_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 1;
    int target = 3;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 3_to_1_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 3;
    int target = 1;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 2_to_3_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 2;
    int target = 3;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, 3_to_2_process_send) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    int sender = 3;
    int target = 2;
    if (sender < world.size() && target < world.size()) {
      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, random_send_between_0_and_max_small_data) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    if (world.size() > 1) {
      int sender;
      int target;
      if (world.rank() == 0) {
        tyshkevich_a_hypercube_mpi::getRandomIntPair(0, world.size() - 1, sender, target);
      }
      boost::mpi::broadcast(world, sender, 0);
      boost::mpi::broadcast(world, target, 0);

      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 10);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, random_send_between_0_and_max_medium_data) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    if (world.size() > 1) {
      int sender;
      int target;
      if (world.rank() == 0) {
        tyshkevich_a_hypercube_mpi::getRandomIntPair(0, world.size() - 1, sender, target);
      }
      boost::mpi::broadcast(world, sender, 0);
      boost::mpi::broadcast(world, target, 0);

      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 100);
    } else {
      GTEST_SKIP();
    }
  }
}

TEST(tyshkevich_a_hypercube_mpi, random_send_between_0_and_max_large_data) {
  boost::mpi::communicator world;
  if (tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    if (world.size() > 1) {
      int sender;
      int target;
      if (world.rank() == 0) {
        tyshkevich_a_hypercube_mpi::getRandomIntPair(0, world.size() - 1, sender, target);
      }
      boost::mpi::broadcast(world, sender, 0);
      boost::mpi::broadcast(world, target, 0);

      tyshkevich_a_hypercube_mpi::run_test(world, sender, target, 1000);
    } else {
      GTEST_SKIP();
    }
  }
}
