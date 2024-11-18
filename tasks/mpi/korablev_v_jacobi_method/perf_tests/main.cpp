#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/korablev_v_jacobi_method/include/ops_mpi.hpp"

std::pair<std::vector<double>, std::vector<double>> generate_diagonally_dominant_matrix(int n, double min_val = -10.0,
                                                                                        double max_val = 10.0) {
  std::vector<double> A(n * n);
  std::vector<double> b(n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A[i * n + j] = dist(gen);
        row_sum += std::abs(A[i * n + j]);
      }
    }

    A[i * n + i] = row_sum + std::abs(dist(gen)) + 1.0;

    b[i] = dist(gen);
  }

  return {A, b};
}

TEST(korablev_v_jacobi_method, test_pipeline_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 1500;
  auto [A_flat, b] = generate_diagonally_dominant_matrix(matrix_size);

  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto jacobiTaskParallel = std::make_shared<korablev_v_jacobi_method_mpi::JacobiMethodParallel>(taskDataPar);
  ASSERT_EQ(jacobiTaskParallel->validation(), true);
  jacobiTaskParallel->pre_processing();
  jacobiTaskParallel->run();
  jacobiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(matrix_size, out.size());
  }
}

TEST(korablev_v_jacobi_method, test_task_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 1500;
  auto [A_flat, b] = generate_diagonally_dominant_matrix(matrix_size);

  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto jacobiTaskParallel = std::make_shared<korablev_v_jacobi_method_mpi::JacobiMethodParallel>(taskDataPar);
  ASSERT_EQ(jacobiTaskParallel->validation(), true);
  jacobiTaskParallel->pre_processing();
  jacobiTaskParallel->run();
  jacobiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(matrix_size, out.size());
  }
}