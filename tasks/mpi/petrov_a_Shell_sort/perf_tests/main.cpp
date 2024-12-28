#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

std::vector<int> generate_random_vector(int n, int min_val = -100, int max_val = 100,
                                        unsigned seed = std::random_device{}()) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(min_val, max_val);

  std::vector<int> vec(n);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
  return vec;
}

TEST(petrov_a_Shell_sort_mpi, pipeline_run) {
  boost::mpi::communicator world;
  int size = 7000000;
  std::vector<int> data;
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data = generate_random_vector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataPar->inputs_count.emplace_back(data.size());
    result_data.resize(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  auto taskParallel = std::make_shared<petrov_a_Shell_sort_mpi::TestTaskMPI>(taskDataPar);

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
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}

TEST(petrov_a_Shell_sort_mpi, task_run) {
  boost::mpi::communicator world;
  int size = 7000000;
  std::vector<int> data;
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data = generate_random_vector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataPar->inputs_count.emplace_back(data.size());
    result_data.resize(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  auto taskParallel = std::make_shared<petrov_a_Shell_sort_mpi::TestTaskMPI>(taskDataPar);

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
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}
