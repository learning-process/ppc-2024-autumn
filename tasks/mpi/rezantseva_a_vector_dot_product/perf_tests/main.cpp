// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/rezantseva_a_vector_dot_product/include/ops_mpi.hpp"

TEST(rezantseva_a_vector_dot_product_mpi, test_pipeline_run) {
  const int count_size_vector = 20000000;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int> v1 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    
    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel = std::make_shared<rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(rezantseva_a_vector_dot_product_mpi::vectorDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}

TEST(rezantseva_a_vector_dot_product_mpi, test_task_run) {
  const int count_size_vector = 20000000;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int> v1 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel = std::make_shared<rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(rezantseva_a_vector_dot_product_mpi::vectorDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}
