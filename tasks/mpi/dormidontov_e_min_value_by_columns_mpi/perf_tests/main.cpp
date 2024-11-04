#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/dormidontov_e_min_value_by_columns_mpi/include/ops_mpi.hpp"

TEST(dormidontov_e_min_value_by_columns_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  int rs = 777;
  int cs = 777;

  std::vector<int> matrix(cs * rs);
  std::vector<int> res_out_paral(cs, 0);
  std::vector<int> exp_res_paral(cs, 0);

  for (int i = 0; i < rs; i++) {
    for (int j = 0; j < cs; j++) {
      if (i == 0) {
        matrix[i * rs + j] = 0;
        continue;
      }
      matrix[i * rs + j] = rand() % 1000;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs);
    taskDataPar->inputs_count.emplace_back(cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  auto testMpiTaskParallel = std::make_shared<dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(res_out_paral, exp_res_paral);
  }
}

TEST(dormidontov_e_min_value_by_columns_mpi, test_task_run) {
  boost::mpi::communicator world;
  int rs = 777;
  int cs = 777;

  std::vector<int> matrix(cs * rs);
  std::vector<int> res_out_paral(cs, 0);
  std::vector<int> exp_res_paral(cs, 0);

  for (int i = 0; i < rs; i++) {
    for (int j = 0; j < cs; j++) {
      if (i == 0) {
        matrix[i * rs + j] = 0;
        continue;
      }
      matrix[i * rs + j] = rand() % 1000;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs);
    taskDataPar->inputs_count.emplace_back(cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  auto testMpiTaskParallel = std::make_shared<dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(res_out_paral, exp_res_paral);
  }
}