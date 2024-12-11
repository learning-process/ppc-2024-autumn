
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/budazhapova_e_matrix_multiplication/include/matrix_mult_mpi.hpp"

TEST(budazhapova_e_matrix_mult_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> A_matrix(8000000, 2);
  std::vector<int> b_vector(2000, 3);
  std::vector<int> out(4000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->outputs_count.emplace_back(b_vector.size());
  }

  auto testMpiTaskParallel = std::make_shared<budazhapova_e_matrix_mult_mpi::MatrixMultParallel>(taskDataPar);
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
  }
}

TEST(budazhapova_e_matrix_mult_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> A_matrix(8000000, 2);
  std::vector<int> b_vector(2000, 3);
  std::vector<int> out(4000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
    taskDataPar->inputs_count.emplace_back(A_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
    taskDataPar->outputs_count.emplace_back(b_vector.size());
  }

  auto testMpiTaskParallel = std::make_shared<budazhapova_e_matrix_mult_mpi::MatrixMultParallel>(taskDataPar);
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
  }
}
