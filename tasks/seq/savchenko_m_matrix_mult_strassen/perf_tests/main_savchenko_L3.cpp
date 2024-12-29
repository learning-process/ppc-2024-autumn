#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/savchenko_m_matrix_mult_strassen/include/ops_seq_savchenko_L3.hpp"

namespace savchenko_m_matrix_mult_strassen_seq {
std::vector<double> getRandomMatrix(size_t size, double min, double max) {
  if (size <= 0) {
    throw std::out_of_range("size must be greater than 0");
  }
  if (min > max) {
    throw std::invalid_argument("min should not be greater than max");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);

  // Forming a random matrix
  std::vector<double> matrix(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      matrix[i * size + j] = dist(gen);
    }
  }

  return matrix;
}

double getRandomDouble(double min, double max) {
  if (min > max) {
    throw std::invalid_argument("min should not be greater than max");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);

  return dist(gen);
}

}  // namespace savchenko_m_matrix_mult_strassen_seq

TEST(savchenko_m_matrix_mult_strassen_seq, test_pipeline_run) {
  // Create data
  const size_t size = 256;

  const double gen_min = -10.0;
  const double gen_max = 10.0;

  std::vector<double> matrix_A = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_B = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  auto testTaskSequential = std::make_shared<savchenko_m_matrix_mult_strassen_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(savchenko_m_matrix_mult_strassen_seq, test_task_run) {
  // Create data
  const size_t size = 256;

  const double gen_min = -10.0;
  const double gen_max = 10.0;

  std::vector<double> matrix_A = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_B = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  auto testTaskSequential = std::make_shared<savchenko_m_matrix_mult_strassen_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
