// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/leontev_n_vector_sum/include/ops_mpi.hpp"

inline void taskEmplacement(std::shared_ptr<ppc::core::TaskData>& taskDataPar, std::vector<int>& global_vec,
                            std::vector<int32_t>& global_sum) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
  taskDataPar->outputs_count.emplace_back(global_sum.size());
}

TEST(leontev_n_vec_sum_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 30000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  auto MPIVecSumParallel = std::make_shared<leontev_n_vec_sum_mpi::MPIVecSumParallel>(taskDataPar);
  ASSERT_EQ(MPIVecSumParallel->validation(), true);
  MPIVecSumParallel->pre_processing();
  MPIVecSumParallel->run();
  MPIVecSumParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIVecSumParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_sum[0]);
  }
}

TEST(leontev_n_vec_sum_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 30000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  auto MPIVecSumParallel = std::make_shared<leontev_n_vec_sum_mpi::MPIVecSumParallel>(taskDataPar);
  ASSERT_EQ(MPIVecSumParallel->validation(), true);
  MPIVecSumParallel->pre_processing();
  MPIVecSumParallel->run();
  MPIVecSumParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIVecSumParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_sum[0]);
  }
}
