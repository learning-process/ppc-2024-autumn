#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_A = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_B = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_C = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_A[dimension * 101] = 1;
    input_matrix_B[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_A.size());
    taskDataPar->inputs_count.emplace_back(input_matrix_B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
    taskDataPar->outputs_count.emplace_back(out_matrix_C.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(input_matrix_A, out_matrix_C[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_A = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_B = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_C = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_A[dimension * 101] = 1;
    input_matrix_B[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_A.size());
    taskDataPar->inputs_count.emplace_back(input_matrix_B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
    taskDataPar->outputs_count.emplace_back(out_matrix_C.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(input_matrix_A, out_matrix_C[0]);
  }
}
