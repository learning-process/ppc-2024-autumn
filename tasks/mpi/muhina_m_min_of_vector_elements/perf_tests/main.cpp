// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/muhina_m_min_of_vector_elements/include/ops_mpi.hpp"

TEST(muhina_m_min_of_vector_elements_mpi, run_pipeline) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000000;
    const int min_val = 0;
    const int max_val = 100;
    global_vec = muhina_m_min_of_vector_elements_mpi::GetRandomVector(count_size_vector, min_val, max_val);
    global_vec[0] = -100;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MinOfVectorMPIParallel =
      std::make_shared<muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel>(taskDataPar);
  ASSERT_EQ(MinOfVectorMPIParallel->validation(), true);
  MinOfVectorMPIParallel->pre_processing();
  MinOfVectorMPIParallel->run();
  MinOfVectorMPIParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MinOfVectorMPIParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(-100, global_res[0]);
  }
}

TEST(muhina_m_min_of_vector_elements_mpi, run_task) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000000;
    const int min_val = 0;
    const int max_val = 100;
    global_vec = muhina_m_min_of_vector_elements_mpi::GetRandomVector(count_size_vector, min_val, max_val);
    global_vec[0] = -100;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MinOfVectorMPIParallel =
      std::make_shared<muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel>(taskDataPar);
  ASSERT_EQ(MinOfVectorMPIParallel->validation(), true);
  MinOfVectorMPIParallel->pre_processing();
  MinOfVectorMPIParallel->run();
  MinOfVectorMPIParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MinOfVectorMPIParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(-100, global_res[0]);
  }
}
