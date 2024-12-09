#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kovalchuk_a_horizontal_tape_scheme/include/ops_seq.hpp"

using namespace kovalchuk_a_horizontal_tape_scheme_seq;

std::vector<int> getRandomVectora(int sz, int min = MINIMALGEN, int max = MAXIMUMGEN);
std::vector<std::vector<int>> getRandomMatrixa(int rows, int columns, int min = MINIMALGEN, int max = MAXIMUMGEN);

std::vector<int> getRandomVectora(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrixa(int rows, int columns, int min, int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVectora(columns, min, max);
  }
  return vec;
}

TEST(kovalchuk_a_horizontal_tape_scheme, test_pipeline_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result(1000, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 1000;
  int count_columns = 1000;
  global_matrix = getRandomMatrixa(count_rows, count_columns);
  global_vector = getRandomVectora(count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());
  // Create Task
  auto testSequentialTask = std::make_shared<kovalchuk_a_horizontal_tape_scheme_seq::TestSequentialTask>(taskDataSeq);
  ASSERT_EQ(testSequentialTask->validation(), true);
  testSequentialTask->pre_processing();
  testSequentialTask->run();
  testSequentialTask->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
        .count();
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSequentialTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(kovalchuk_a_horizontal_tape_scheme, test_task_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 3;
  int count_columns = 3;
  global_matrix = getRandomMatrixa(count_rows, count_columns);
  global_vector = getRandomVectora(count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());
  // Create Task
  auto testSequentialTask = std::make_shared<kovalchuk_a_horizontal_tape_scheme_seq::TestSequentialTask>(taskDataSeq);
  ASSERT_EQ(testSequentialTask->validation(), true);
  testSequentialTask->pre_processing();
  testSequentialTask->run();
  testSequentialTask->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
        .count();
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSequentialTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
