#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_mpi.hpp"

namespace sozonov_i_gaussian_method_horizontal_strip_scheme_mpi {

std::vector<double> getRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> mat(sz);
  for (int i = 0; i < sz; ++i) {
    mat[i] = dis(gen);
  }
  return mat;
}

double Ax_b(int n, int m, std::vector<double> a, std::vector<double> x) {
  std::vector<double> c(m, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n - 1; ++j) {
      c[i] += a[i * n + j] * x[j];
    }
    c[i] -= a[i * n + m];
  }

  double c_norm = 0;
  for (int i = 0; i < m; i++) {
    c_norm += c[i] * c[i];
  }
  return sqrt(c_norm);
}

}  // namespace sozonov_i_gaussian_method_horizontal_strip_scheme_mpi

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const double EPS = 1e-6;

  const int cols = 101;
  const int rows = 100;
  std::vector<double> global_mat(cols * rows);
  std::vector<double> global_ans(cols - 1, 0);
  double ans;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::getRandomMatrix(cols * rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
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
    ans = sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::Ax_b(cols, rows, global_mat, global_ans);
    ASSERT_NEAR(ans, 0, EPS);
  }
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_task_run) {
  boost::mpi::communicator world;

  const double EPS = 1e-6;

  const int cols = 101;
  const int rows = 100;
  std::vector<double> global_mat(cols * rows);
  std::vector<double> global_ans(cols - 1, 0);
  double ans;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::getRandomMatrix(cols * rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
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
    ans = sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::Ax_b(cols, rows, global_mat, global_ans);
    ASSERT_NEAR(ans, 0, EPS);
  }
}
