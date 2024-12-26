#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/tyshkevich_a_hypercube/include/ops_mpi.hpp"

namespace tyshkevich_a_hypercube_mpi {

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

bool initial_test(int world_size) {
  int init_dimension = static_cast<int>(std::log2(world_size));
  return world_size > 1 && (1 << init_dimension) == world_size;
}

}  // namespace tyshkevich_a_hypercube_mpi

TEST(tyshkevich_a_hypercube_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (!tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    GTEST_SKIP();
  }

  int sender = 0;
  int target = world.size() - 1;
  int data_size = 1000;

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

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }

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

TEST(tyshkevich_a_hypercube_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (!tyshkevich_a_hypercube_mpi::initial_test(world.size())) {
    GTEST_SKIP();
  }

  int sender = 0;
  int target = world.size() - 1;
  int data_size = 1000;

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

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }

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
