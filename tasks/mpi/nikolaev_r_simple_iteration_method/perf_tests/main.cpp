#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/nikolaev_r_simple_iteration_method/include/ops_mpi.hpp"

std::pair<std::vector<double>, std::vector<double>> generate_random_diagonally_dominant_matrix_and_free_terms(
    size_t size, double min_val, double max_val) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<> distribution(min_val, max_val);

  std::vector<double> matrix(size * size);
  std::vector<double> free_terms(size);

  for (size_t row = 0; row < size; ++row) {
    double sum_non_diagonal = 0.0;

    for (size_t col = 0; col < size; ++col) {
      if (row != col) {
        matrix[row * size + col] = distribution(generator);
        sum_non_diagonal += std::abs(matrix[row * size + col]);
      }
    }

    matrix[row * size + row] = sum_non_diagonal + std::abs(distribution(generator)) + 1.0;

    free_terms[row] = distribution(generator);
  }

  return {matrix, free_terms};
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const size_t m_size = 500;
  auto [A, b] = generate_random_diagonally_dominant_matrix_and_free_terms(m_size, -15.0, 15.0);
  std::vector<size_t> in(1, m_size);
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel =
      std::make_shared<nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel>(taskDataPar);
  ASSERT_TRUE(testTaskParallel->validation());
  ASSERT_TRUE(testTaskParallel->pre_processing());
  ASSERT_TRUE(testTaskParallel->run());
  ASSERT_TRUE(testTaskParallel->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(m_size, out.size());
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_task_run) {
  boost::mpi::communicator world;

  const size_t m_size = 500;
  auto [A, b] = generate_random_diagonally_dominant_matrix_and_free_terms(m_size, -15.0, 15.0);
  std::vector<size_t> in(1, m_size);
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel =
      std::make_shared<nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel>(taskDataPar);
  ASSERT_TRUE(testTaskParallel->validation());
  ASSERT_TRUE(testTaskParallel->pre_processing());
  ASSERT_TRUE(testTaskParallel->run());
  ASSERT_TRUE(testTaskParallel->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(m_size, out.size());
  }
}