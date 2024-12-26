#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/matthew_fyodorov_reduce/include/ops_mpi.hpp"

TEST(matthew_fyodorov_reduce_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int size = world.size();
  const int vector_size = 10000000;
  std::vector<int> global_vector(vector_size, 0);
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-100, 100);
    for (int i = 0; i < vector_size; i++) {
      global_vector[i] = dist(gen);
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<matthew_fyodorov_reduce_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    ASSERT_EQ(1, 1);
  }
}

TEST(matthew_fyodorov_reduce_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int rank = world.rank();
  int size = world.size();
  const int vector_size = 100000;
  std::vector<int> global_vector;
  std::vector<int32_t> global_sum(1, 0);

  if (rank == 0) {
    global_vector.resize(vector_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-100, 100);
    for (int i = 0; i < vector_size; ++i) {
      global_vector[i] = dist(gen);
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<matthew_fyodorov_reduce_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    if (rank == 0) {
      ASSERT_EQ(1, 1);
    }
  }
}
