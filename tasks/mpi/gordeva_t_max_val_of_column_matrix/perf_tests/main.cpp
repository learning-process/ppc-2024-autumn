#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <climits>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gordeva_t_max_val_of_column_matrix/include/ops_mpi.hpp"

std::vector<int> rand_vec_1(int s, int down, int upp) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(down, upp);

  std::vector<int> v(s);
  for (auto& number : v) {
    number = dis(gen);
  }
  return v;
}

std::vector<std::vector<int>> rand_matr_1(int rows, int cols) {
  std::vector<std::vector<int>> matr(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; ++i) {
    matr[i] = rand_vec_1(cols, 0, 100);
  }
  for (int j = 0; j < cols; ++j) {
    int row_rand = std::rand() % rows;
    matr[row_rand][j] = 10;
  }
  return matr;
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int cols = 5000;
  const int rows = 5000;
  std::vector<int> res_vec_par(cols, 0);
  std::vector<int> res_vec(cols, 0);
  std::vector<std::vector<int>> matrix;

  std::vector<std::vector<int>> matrix_1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = rand_matr_1(rows, cols);
    for (auto& i : matrix) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(i.data()));

    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(res_vec_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  matrix_1 = matrix;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    for (size_t i = 0; i < cols; i++) {
      ASSERT_EQ(100, res_vec_par[i]);
    }
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int cols = 5000;
  const int rows = 5000;
  std::vector<int> res_vec_par(cols, 0);
  std::vector<int> res_vec(cols, 0);
  std::vector<std::vector<int>> matrix;

  std::vector<std::vector<int>> matrix_1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = rand_matr_1(rows, cols);
    for (auto& i : matrix) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(i.data()));

    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(res_vec_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  matrix_1 = matrix;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    for (size_t i = 0; i < cols; i++) {
      ASSERT_EQ(100, res_vec_par[i]);
    }
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
