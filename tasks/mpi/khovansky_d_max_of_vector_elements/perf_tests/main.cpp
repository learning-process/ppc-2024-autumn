// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/khovansky_d_max_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> khovansky_d_max_of_vector_elements_mpi::GetRandomVector(int sz, int left, int right) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(sz);
  for (int i = 0; i < sz; i++) {
    v[i] = gen() % (1 + right - left) + left;
  }
  return v;
}

TEST(khovansky_d_max_of_vector_elements_mpi, run_pipeline) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000000;
    const int left = 0;
    const int right = 100;
    global_vec = khovansky_d_max_of_vector_elements_mpi::GetRandomVector(count_size_vector, left, right);
    global_vec[0] = 101;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MaxOfVectorMPIParallel =
      std::make_shared<khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel>(taskDataPar);
  ASSERT_EQ(MaxOfVectorMPIParallel->validation(), true);
  MaxOfVectorMPIParallel->pre_processing();
  MaxOfVectorMPIParallel->run();
  MaxOfVectorMPIParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MaxOfVectorMPIParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(101, global_res[0]);
  }
}

TEST(khovansky_d_max_of_vector_elements_mpi, run_task) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000000;
    const int left = 0;
    const int right = 100;
    global_vec = khovansky_d_max_of_vector_elements_mpi::GetRandomVector(count_size_vector, left, right);
    global_vec[0] = 101;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MaxOfVectorMPIParallel =
      std::make_shared<khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel>(taskDataPar);
  ASSERT_EQ(MaxOfVectorMPIParallel->validation(), true);
  MaxOfVectorMPIParallel->pre_processing();
  MaxOfVectorMPIParallel->run();
  MaxOfVectorMPIParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MaxOfVectorMPIParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(101, global_res[0]);
  }
}
