#include <gtest/gtest.h>

#include <chrono>
#include <random>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_horizontal_line_filtration/include/ops_seq.h"

namespace sharamygina_i_horizontal_line_filtration_seq {
std::vector<unsigned int> GetImage(int rows, int cols) {
  std::vector<unsigned int> temporaryIm(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) temporaryIm[i * cols + j] = dist(gen);
  return temporaryIm;
}

}  // namespace sharamygina_i_horizontal_line_filtration_seq

#define PERF_TEST_SEQ(test_name, rows_const, cols_const, num_runs, perf_method)
TEST(sharamygina_i_horizontal_line_filtration_seq, test_name) {
  int rows = rows_const;
  int cols = cols_const;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<unsigned int> input = sharamygina_i_horizontal_line_filtration_seq::GetImage(rows, cols);
  std::vector<unsigned int> output(rows * cols);
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(rows);
  taskData->inputs_count.push_back(cols);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.push_back(output.size());
  auto task = std::make_shared<sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = num_runs;
  auto start = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->perf_method(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

PERF_TEST_SEQ(LargeImage, 5000, 5000, 1, pipeline_run)

PERF_TEST_SEQ(LargeImageRun, 5000, 5000, 1, task_run)

#undef PERF_TEST_SEQ