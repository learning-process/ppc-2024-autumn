// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/leontev_n_gather/include/ops_mpi.hpp"

inline void taskEmplacement(std::shared_ptr<ppc::core::TaskData>& taskDataPar, std::vector<int>& global_vec,
                            std::vector<int>& global_mat, std::vector<int32_t>& global_res) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
  taskDataPar->inputs_count.emplace_back(global_mat.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
  taskDataPar->outputs_count.emplace_back(global_res.size());
}

TEST(leontev_n_mat_vec_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int count_size_vector = 500;
  std::vector<int> global_vec(count_size_vector, 1);
  std::vector<int> global_mat(count_size_vector * count_size_vector, 1);
  std::vector<int> global_res(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, global_vec, global_mat, global_res);
  }
  auto MPIMatVecParallel = std::make_shared<leontev_n_mat_vec_mpi::MPIMatVecParallel>(taskDataPar);
  ASSERT_EQ(MPIMatVecParallel->validation(), true);
  MPIMatVecParallel->pre_processing();
  MPIMatVecParallel->run();
  MPIMatVecParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIMatVecParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < count_size_vector; i++) {
      ASSERT_EQ(count_size_vector, global_res[i]);
    }
  }
}

TEST(leontev_n_mat_vec_mpi, test_task_run) {
  boost::mpi::communicator world;
  int count_size_vector = 500;
  std::vector<int> global_vec(count_size_vector, 1);
  std::vector<int> global_mat(count_size_vector * count_size_vector, 1);
  std::vector<int> global_res(count_size_vector);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = std::vector<int>(count_size_vector, 1);
    taskEmplacement(taskDataPar, global_vec, global_mat, global_res);
  }
  auto MPIMatVecParallel = std::make_shared<leontev_n_mat_vec_mpi::MPIMatVecParallel>(taskDataPar);
  ASSERT_EQ(MPIMatVecParallel->validation(), true);
  MPIMatVecParallel->pre_processing();
  MPIMatVecParallel->run();
  MPIMatVecParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIMatVecParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < count_size_vector; i++) {
      ASSERT_EQ(count_size_vector, global_res[i]);
    }
  }
}
