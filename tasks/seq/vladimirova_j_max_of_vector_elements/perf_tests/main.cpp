#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vladimirova_j_max_of_vector_elements/include/ops_seq.hpp"

std::vector<int> CreateVector(size_t size, size_t spread_of_val) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> v(size);
  for (size_t i = 0; i < size; i++) {
    v[i] = (random() % (2 * spread_of_val + 1)) - spread_of_val;
  }
  return v;
}

std::vector<std::vector<int>> CreateInputMatrix(size_t row_c, size_t column_c, size_t spread_of_val) {
  //  Init value for input and output
  std::vector<std::vector<int>> m(row_c);
  for (size_t i = 0; i < row_c; i++) {
    m[i] = CreateVector(column_c, spread_of_val);
  }
  return m;
}

TEST(vladimirova_j_max_of_vector_elements_seq, test_pipeline_run) {
  std::random_device dev;
  std::mt19937 random(dev());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int size = 7000;
  int spread = 7000;

  std::vector<std::vector<int>> matrix_in;
  matrix_in = CreateInputMatrix(size, size, spread);
  std::vector<int32_t> out(1, matrix_in[0][0]);

  int some_row = random() % size;
  int some_col = random() % size;
  matrix_in[some_row][some_col] = spread;

  for (unsigned int i = 0; i < matrix_in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<vladimirova_j_max_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(spread, out[0]);
}

TEST(sequential_vladimirova_j_max_of_vector_elements_seq, test_task_run) {
  std::random_device dev;
  std::mt19937 random(dev());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int size = 7000;
  int spread = 7000;
  std::vector<std::vector<int>> matrix_in;
  matrix_in = CreateInputMatrix(size, size, spread);
  std::vector<int32_t> out(1, matrix_in[0][0]);

  int some_row = random() % size;
  int some_col = random() % size;
  matrix_in[some_row][some_col] = spread;

  for (unsigned int i = 0; i < matrix_in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<vladimirova_j_max_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(spread, out[0]);
}
