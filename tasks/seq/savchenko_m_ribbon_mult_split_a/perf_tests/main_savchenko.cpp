#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/savchenko_m_ribbon_mult_split_a/include/ops_seq_savchenko.hpp"

namespace savchenko_m_ribbon_mult_split_a_seq {
std::vector<int> getRandomMatrix(size_t rows, size_t columns, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());

  // Forming a random matrix
  std::vector<int> matrix(rows * columns);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      matrix[i * columns + j] = min + gen() % (max - min + 1);
    }
  }

  return matrix;
}

int getRandomInt(int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int rand_int = min + gen() % (max - min + 1);
  return rand_int;
}
}  // namespace savchenko_m_ribbon_mult_split_a_seq

TEST(savchenko_m_ribbon_mult_split_a_seq, test_pipeline_run) {
  // Create data
  const int fund_size = 516;
  const int columns_A = fund_size;
  const int rows_A = fund_size;
  const int columns_B = fund_size;
  const int rows_B = fund_size;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;

  std::vector<int> matrix_A = savchenko_m_ribbon_mult_split_a_seq::getRandomMatrix(rows_A, columns_A, gen_min, gen_max);
  std::vector<int> matrix_B = savchenko_m_ribbon_mult_split_a_seq::getRandomMatrix(rows_B, columns_B, gen_min, gen_max);
  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential>(taskDataSeq);

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

TEST(savchenko_m_ribbon_mult_split_a_seq, test_task_run) {
  // Create data
  const int fund_size = 516;
  const int columns_A = fund_size;
  const int rows_A = fund_size;
  const int columns_B = fund_size;
  const int rows_B = fund_size;
  const int res_size = rows_A * columns_B;

  const int gen_min = -1000;
  const int gen_max = 1000;

  std::vector<int> matrix_A = savchenko_m_ribbon_mult_split_a_seq::getRandomMatrix(rows_A, columns_A, gen_min, gen_max);
  std::vector<int> matrix_B = savchenko_m_ribbon_mult_split_a_seq::getRandomMatrix(rows_B, columns_B, gen_min, gen_max);
  std::vector<int> matrix_res(res_size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //// matrix_A
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs_count.emplace_back(columns_A);
  taskDataSeq->inputs_count.emplace_back(rows_A);
  //// matrix_B
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(columns_B);
  taskDataSeq->inputs_count.emplace_back(rows_B);
  //// matrix_res
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential>(taskDataSeq);

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
