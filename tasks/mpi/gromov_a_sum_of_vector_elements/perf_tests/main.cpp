#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gromov_a_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(gromov_a_sum_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 50000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto MPISumOfVectorParallel =
      std::make_shared<gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel>(taskDataPar, "add");
  ASSERT_EQ(MPISumOfVectorParallel->validation(), true);
  MPISumOfVectorParallel->pre_processing();
  MPISumOfVectorParallel->run();
  MPISumOfVectorParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPISumOfVectorParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_sum[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 50000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto MPISumOfVectorParallel =
      std::make_shared<gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel>(taskDataPar, "add");
  ASSERT_EQ(MPISumOfVectorParallel->validation(), true);
  MPISumOfVectorParallel->pre_processing();
  MPISumOfVectorParallel->run();
  MPISumOfVectorParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPISumOfVectorParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_sum[0]);
  }
}
