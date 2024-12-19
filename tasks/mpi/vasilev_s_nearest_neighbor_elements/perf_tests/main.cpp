#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vasilev_s_nearest_neighbor_elements/include/ops_mpi.hpp"

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 1000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(vasilev_s_nearest_neighbor_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_result(3, 0);  // min_diff, index1, index2

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 1000000;
    global_vec = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI>(taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
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

    std::vector<int> reference_result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    auto taskSequential =
        std::make_shared<vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI>(taskDataSeq);
    ASSERT_EQ(taskSequential->validation(), true);
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result[0], reference_result[0]);  // min_diff
    ASSERT_EQ(global_result[1], reference_result[1]);  // index1
    ASSERT_EQ(global_result[2], reference_result[2]);  // index2
  }
}

TEST(vasilev_s_nearest_neighbor_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_result(3, 0);  // min_diff, index1, index2

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 1000000;
    global_vec = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI>(taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
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

    std::vector<int> reference_result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    auto taskSequential =
        std::make_shared<vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI>(taskDataSeq);
    ASSERT_EQ(taskSequential->validation(), true);
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result[0], reference_result[0]);  // min_diff
    ASSERT_EQ(global_result[1], reference_result[1]);  // index1
    ASSERT_EQ(global_result[2], reference_result[2]);  // index2
  }
}