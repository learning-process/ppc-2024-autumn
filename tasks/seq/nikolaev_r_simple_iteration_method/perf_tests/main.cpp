#include <gtest/gtest.h>

#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/nikolaev_r_simple_iteration_method/include/ops_seq.hpp"

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

TEST(nikolaev_r_simple_iteration_method_seq, test_pipeline_run) {
  const size_t m_size = 500;
  auto [A, b] = generate_random_diagonally_dominant_matrix_and_free_terms(m_size, -15.0, 15.0);

  std::vector<size_t> in(1, m_size);
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential>(taskDataSeq);
  ASSERT_TRUE(testTaskSequential->validation());
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(m_size, out.size());
}

TEST(nikolaev_r_simple_iteration_method_seq, test_task_run) {
  const size_t m_size = 500;
  auto [A, b] = generate_random_diagonally_dominant_matrix_and_free_terms(m_size, -15.0, 15.0);

  std::vector<size_t> in(1, m_size);
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential>(taskDataSeq);
  ASSERT_TRUE(testTaskSequential->validation());
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(m_size, out.size());
}