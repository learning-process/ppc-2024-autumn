#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_mpi.hpp"

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int count = 1000;
  std::vector<double> global_mat(count * (count + 1), 0);
  std::vector<double> global_ans(count, 0);
  std::vector<double> ans(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < count; ++i) {
      global_mat[i * (count + 1) + i] = 1;
      global_mat[i * (count + 1) + count] = i + 1;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(ans, global_ans);
  }
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int count = 1000;
  std::vector<double> global_mat(count * (count + 1), 0);
  std::vector<double> global_ans(count, 0);
  std::vector<double> ans(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < count; ++i) {
      global_mat[i * (count + 1) + i] = 1;
      global_mat[i * (count + 1) + count] = i + 1;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(ans, global_ans);
  }
}
