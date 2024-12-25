#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vavilov_v_min_elements_in_columns_of_matrix/include/ops_seq.hpp"

std::vector<int> generate_rand_vec(int size, int lower_bound, int upper_bound) {
  std::vector<int> vec(size);
  for (auto& n : vec) {
    n = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return vec;
}

std::vector<std::vector<int>> generate_rand_matr(int rows, int cols) {
  std::vector<std::vector<int>> matr(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; i++) {
    matr[i] = generate_rand_vec(cols, -1000, 1000);
  }
  for (int j = 0; j < cols; j++) {
    int r_row = std::rand() % rows;
    matr[r_row][j] = INT_MIN;
  }
  return matr;
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, test_pipeline_run) {
  const int rows = 5000;
  const int cols = 5000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential =
      std::make_shared<vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential>(taskDataSeq);

  std::vector<std::vector<int>> matr = generate_rand_matr(rows, cols);

  for (auto& row : matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Set the number of runs as needed
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

  for (size_t i = 0; i < cols; i++) {
    ASSERT_EQ(vec_res[i], INT_MIN);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, test_task_run) {
  const int rows = 5000;
  const int cols = 5000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential =
      std::make_shared<vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential>(taskDataSeq);

  std::vector<std::vector<int>> matr = generate_rand_matr(rows, cols);

  for (auto& row : matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

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

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(vec_res[i], INT_MIN);
  }
}
