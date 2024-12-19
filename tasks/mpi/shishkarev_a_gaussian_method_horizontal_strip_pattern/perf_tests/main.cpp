#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

std::vector<double> getRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> matrix(sz);
  for (int i = 0; i < sz; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

double Ax_b(int n, int m, std::vector<double> a, std::vector<double> res) {
  std::vector<double> tmp(m, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n - 1; ++j) {
      tmp[i] += a[i * n + j] * res[j];
    }
    tmp[i] -= a[i * n + m];
  }

  double tmp_norm = 0;
  for (int i = 0; i < m; i++) {
    tmp_norm += tmp[i] * tmp[i];
  }
  return sqrt(tmp_norm);
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int cols = 101;
  const int rows = 100;
  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::getRandomMatrix(cols * rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MPIGaussHorizontalParallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel>(
          taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel->validation(), true);
  MPIGaussHorizontalParallel->pre_processing();
  MPIGaussHorizontalParallel->run();
  MPIGaussHorizontalParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussHorizontalParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::Ax_b(cols, rows, global_matrix, global_res),
                0, 1e-6);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int cols = 101;
  const int rows = 100;
  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::getRandomMatrix(cols * rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MPIGaussHorizontalParallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel>(
          taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel->validation(), true);
  MPIGaussHorizontalParallel->pre_processing();
  MPIGaussHorizontalParallel->run();
  MPIGaussHorizontalParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussHorizontalParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::Ax_b(cols, rows, global_matrix, global_res),
                0, 1e-6);
  }
}