#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/belov_a_max_value_of_matrix_elements/include/ops_seq.hpp"

template <typename T = int>
std::vector<T> generate_random_matrix(int rows, int cols, const T &left = T{-1000}, const T &right = T{1000}) {
  std::vector<T> res(rows * cols);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (size_t i = 0; i < res.size(); i++) {
    res[i] = left + static_cast<T>(gen() % int(right - left + 1));
  }
  return res;
}

TEST(belov_a_max_value_matrix_seq_perf_test, test_pipeline_run) {
  const int rows = 600;
  const int cols = 950;

  // Create data
  std::vector<int> matrix = generate_random_matrix<int>(rows, cols);

  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<belov_a_max_value_of_matrix_elements_seq::MaxValueOfMatrixElementsSequential<int>>(taskDataSeq);

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
  ASSERT_EQ(*std::max_element(matrix.begin(), matrix.end()), out[0]);
}

TEST(belov_a_max_value_matrix_seq_perf_test, test_task_run) {
  const int rows = 600;
  const int cols = 950;

  // Create data
  std::vector<int> matrix = generate_random_matrix<int>(rows, cols);

  std::vector<int> dimensions = {rows, cols};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dimensions.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(dimensions.size());
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<belov_a_max_value_of_matrix_elements_seq::MaxValueOfMatrixElementsSequential<int>>(taskDataSeq);

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
  ASSERT_EQ(*std::max_element(matrix.begin(), matrix.end()), out[0]);
}
