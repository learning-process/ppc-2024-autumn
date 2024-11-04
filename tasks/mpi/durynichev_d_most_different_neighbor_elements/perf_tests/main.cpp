#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/durynichev_d_most_different_neighbor_elements/include/ops_mpi.hpp"

TEST(durynichev_d_most_different_neighbor_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_results(2, 0);
  std::vector<uint64_t> global_indices(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 10000000;
  if (world.rank() == 0) {
    global_vec.resize(count_size_vector, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskData->inputs_count.push_back(global_vec.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
    taskData->outputs_count.push_back(global_results.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
    taskData->outputs_count.push_back(global_indices.size());
  }

  auto task =
      std::make_shared<durynichev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int>>(
          taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(static_cast<size_t>(count_size_vector), global_vec.size());
  }
}

TEST(durynichev_d_most_different_neighbor_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_results(2, 0);
  std::vector<uint64_t> global_indices(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 1000000;
  if (world.rank() == 0) {
    global_vec.resize(count_size_vector, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskData->inputs_count.push_back(global_vec.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
    taskData->outputs_count.push_back(global_results.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
    taskData->outputs_count.push_back(global_indices.size());
  }

  auto task =
      std::make_shared<durynichev_d_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int>>(
          taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(static_cast<size_t>(count_size_vector), global_vec.size());
  }
}
