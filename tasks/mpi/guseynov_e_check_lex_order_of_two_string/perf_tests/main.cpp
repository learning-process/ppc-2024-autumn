#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/guseynov_e_check_lex_order_of_two_string/include/ops_mpi.hpp"

TEST(guseynov_e_check_lex_order_of_two_string_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> global_vec(2, std::vector<char>(25000000, 'a'));
  std::vector<int> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMPITaskParallel =
      std::make_shared<guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMPITaskParallel->validation(), true);
  testMPITaskParallel->pre_processing();
  testMPITaskParallel->run();
  testMPITaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMPITaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(0, global_res[0]);
  }
}

TEST(guseynov_e_check_lex_order_of_two_string_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> global_vec(2, std::vector<char>(25000000, 'a'));
  std::vector<int> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMPITaskParallel =
      std::make_shared<guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMPITaskParallel->validation(), true);
  testMPITaskParallel->pre_processing();
  testMPITaskParallel->run();
  testMPITaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMPITaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(0, global_res[0]);
  }
}