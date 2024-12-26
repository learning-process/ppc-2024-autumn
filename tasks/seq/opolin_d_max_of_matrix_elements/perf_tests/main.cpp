// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/opolin_d_max_of_matrix_elements/include/ops_seq.hpp"

std::vector<int> getRandomVectorForGetMaxInMatrix(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrixForGetMaxInMatrix(int rows, int cols, int min, int max) {
  std::vector<std::vector<int>> matr(rows);
  for (int i = 0; i < rows; i++) {
    matr[i] = getRandomVectorForGetMaxInMatrix(cols, min, max);
  }
  return matr;
}

TEST(opolin_d_max_of_matrix_elements_seq, test_pipeline_run) {
  std::vector<std::vector<int>> matrix;
  std::vector<int32_t> global_out(1, std::numeric_limits<int32_t>::min());

  std::random_device dev;
  std::mt19937 gen(dev());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows = 2000;
  int cols = 2000;
  int min = 0;
  int max = 1000;
  int res = std::numeric_limits<int32_t>::max();

  matrix = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
  int rand_c = gen() % cols;
  int rand_r = gen() % rows;
  matrix[rand_r][rand_c] = res;
  for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
  taskDataSeq->outputs_count.emplace_back(global_out.size());
  auto testTaskSequential = std::make_shared<opolin_d_max_of_matrix_elements_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto start = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res, global_out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, test_task_run) {
  std::vector<std::vector<int>> matrix;
  std::vector<int32_t> global_out(1, std::numeric_limits<int32_t>::min());

  std::random_device dev;
  std::mt19937 gen(dev());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows = 2000;
  int cols = 2000;
  int min = 0;
  int max = 1000;
  int res = std::numeric_limits<int32_t>::max();

  matrix = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
  int rand_c = gen() % cols;
  int rand_r = gen() % rows;
  matrix[rand_r][rand_c] = res;
  for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
  taskDataSeq->outputs_count.emplace_back(global_out.size());
  auto testTaskSequential = std::make_shared<opolin_d_max_of_matrix_elements_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto start = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res, global_out[0]);
}