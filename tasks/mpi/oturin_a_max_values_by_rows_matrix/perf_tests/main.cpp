#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/oturin_a_max_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(oturin_a_max_values_by_rows_matrix_mpi_perftest, test_pipeline_run) {
  size_t n = 300;
  size_t m = 300;

  int argc = 1;
  char **argv;
  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world_rank == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel>(taskDataPar, "max");
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
  if (world_rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ((int)(n * m), global_max[0]);
  }
  MPI_Finalize();
}

TEST(oturin_a_max_values_by_rows_matrix_mpi_perftest, test_task_run) {
  size_t n = 300;
  size_t m = 300;

  int argc = 1;
  char **argv;
  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world_rank == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel>(taskDataPar, "max");
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
  if (world_rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ((int)(n * m), global_max[0]);
  }
  MPI_Finalize();
}
