#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kurakin_m_min_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(kurakin_m_min_values_by_rows_matrix_mpi, test_pipeline_run) {
  const int count_rows = 100;
  const int size_rows = 400;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_mat = std::vector<int>(count_rows * size_rows, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel>(
      taskDataPar, count_rows, size_rows);
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
    for (unsigned i = 0; i < par_min_vec.size(); i++) {
      // std::cout << par_min_vec[0] << " ";
      EXPECT_EQ(1, par_min_vec[0]);
    }
    std::cout << "\n";
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, test_task_run) {
  const int count_rows = 100;
  const int size_rows = 400;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_mat = std::vector<int>(count_rows * size_rows, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel>(
      taskDataPar, count_rows, size_rows);
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
    for (unsigned i = 0; i < par_min_vec.size(); i++) {
      // std::cout << par_min_vec[0] << " ";
      EXPECT_EQ(1, par_min_vec[0]);
    }
    std::cout << "\n";
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
