// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/muhina_m_shell_sort/include/ops_mpi.hpp"

namespace muhina_m_shell_sort_mpi {

std::vector<int> Get_Random_Vector(int sz, int min_value, int max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min_value + gen() % (max_value - min_value + 1);
  }
  return vec;
}
}  // namespace muhina_m_shell_sort_mpi

TEST(muhina_m_shell_sort_mpi, run_pipeline) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;

  if (world.rank() == 0) {
    count_size_vector = 2500000;
    const int min_val = 0;
    const int max_val = 100;
    global_vec = muhina_m_shell_sort_mpi::Get_Random_Vector(count_size_vector, min_val, max_val);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto ShellSortMPIParallel = std::make_shared<muhina_m_shell_sort_mpi::ShellSortMPIParallel>(taskDataPar);
  ASSERT_EQ(ShellSortMPIParallel->validation(), true);
  ShellSortMPIParallel->pre_processing();
  ShellSortMPIParallel->run();
  ShellSortMPIParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortMPIParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> expected_result = global_vec;
    std::sort(expected_result.begin(), expected_result.end());
    ASSERT_EQ(global_res, expected_result);
  }
}

TEST(muhina_m_shell_sort_mpi, run_task) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 2500000;
    const int min_val = 0;
    const int max_val = 100;
    global_vec = muhina_m_shell_sort_mpi::Get_Random_Vector(count_size_vector, min_val, max_val);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto ShellSortMPIParallel = std::make_shared<muhina_m_shell_sort_mpi::ShellSortMPIParallel>(taskDataPar);
  ASSERT_EQ(ShellSortMPIParallel->validation(), true);
  ShellSortMPIParallel->pre_processing();
  ShellSortMPIParallel->run();
  ShellSortMPIParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortMPIParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    std::vector<int> expected_result = global_vec;
    std::sort(expected_result.begin(), expected_result.end());
    ASSERT_EQ(global_res, expected_result);
  }
}
