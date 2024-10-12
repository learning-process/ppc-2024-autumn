
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chistov_a_sum_of_matrix_elements/include/ops_mpi.hpp"

//TEST(mpi_example_perf_test, test_pipeline_run) {
//  boost::mpi::communicator world;
//  std::vector<int> global_matrix;
//  std::vector<int32_t> global_sum(1, 0);
//
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  int count_size_vector;
//  if (world.rank() == 0) {
//    count_size_vector = 120;
//    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(3, 3);
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
//    taskDataPar->inputs_count.emplace_back(global_matrix.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
//    taskDataPar->outputs_count.emplace_back(global_sum.size());
//  }
//
//
//
//  auto testMpiTaskParallel =
//      std::make_shared<chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int>>(taskDataPar, 3, 3);
//
//  ASSERT_EQ(testMpiTaskParallel->validation(), true);
//  ASSERT_EQ(testMpiTaskParallel -> pre_processing(), true);
//  ASSERT_EQ(testMpiTaskParallel -> run(), true);
//  ASSERT_EQ(testMpiTaskParallel->post_processing(), true);
//
//
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 10;
//  const boost::mpi::timer current_timer;
//  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
//
//  // Create and init perf results
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//
//  // Create Perf analyzer
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//  perfAnalyzer->pipeline_run(perfAttr, perfResults);
//  if (world.rank() == 0) {
//    ppc::core::Perf::print_perf_statistic(perfResults);
//    ASSERT_EQ(count_size_vector, global_sum[0]);
//  }
//}

//TEST(mpi_example_perf_test, test_task_run) {
//  boost::mpi::communicator world;
//  std::vector<int> global_matrix;
//  std::vector<int32_t> global_sum(1, 0);
//   Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  int count_size_vector;
//  if (world.rank() == 0) {
//    count_size_vector = 120;
//    global_matrix = std::vector<int>(3, 3);
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
//    taskDataPar->inputs_count.emplace_back(global_matrix.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
//    taskDataPar->outputs_count.emplace_back(global_sum.size());
//  }
//
//  auto testMpiTaskParallel = std::make_shared<chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int>>(taskDataPar,3,3);
//  ASSERT_EQ(testMpiTaskParallel->validation(), true);
//  testMpiTaskParallel->pre_processing();
//  testMpiTaskParallel->run();
//  testMpiTaskParallel->post_processing();
//
//   Create Perf attributes
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 10;
//  const boost::mpi::timer current_timer;
//  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
//
//   Create and init perf results
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//
//   Create Perf analyzer
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//  perfAnalyzer->task_run(perfAttr, perfResults);
//  if (world.rank() == 0) {
//    ppc::core::Perf::print_perf_statistic(perfResults);
//    ASSERT_EQ(count_size_vector, global_sum[0]);
//  }
//}

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
