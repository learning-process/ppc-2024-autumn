#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shkurinskaya_e_gauss_jordan/include/ops_seq.hpp"

namespace shkurinskaya_e_gauss_jordan_seq {

std::vector<double> generate_invertible_matrix(int size) {
  std::vector<double> matrix(size * (size + 1));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    double diag = (i * (size + 1) + i);
    for (int j = 0; j < size + 1; ++j) {
      if (i != j) {
        matrix[i * (size + 1) + j] = dist(gen);
        row_sum += std::abs(matrix[i * (size + 1) + j]);
      }
    }
    matrix[diag] = row_sum + 1;
  }

  return matrix;
}

}  // namespace shkurinskaya_e_gauss_jordan_seq

TEST(shkurinskaya_e_gauss_jordan_seq, test_pipeline_run) {
  size_t size = 500;

  std::vector<double> matrix = shkurinskaya_e_gauss_jordan_seq::generate_invertible_matrix(size);

  std::vector<double> output_data(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  auto gaussTaskSequential = std::make_shared<shkurinskaya_e_gauss_jordan_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_EQ(gaussTaskSequential->validation(), true);
  gaussTaskSequential->pre_processing();
  gaussTaskSequential->run();
  gaussTaskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(output_data.size(), size);
}

TEST(shkurinskaya_e_gauss_jordan, test_task_run) {
  size_t size = 500;

  auto matrix = shkurinskaya_e_gauss_jordan_seq::generate_invertible_matrix(size);

  std::vector<double> output_data(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  auto gaussTaskSequential = std::make_shared<shkurinskaya_e_gauss_jordan_seq::TestTaskSequential>(taskDataSeq);
  ASSERT_EQ(gaussTaskSequential->validation(), true);
  gaussTaskSequential->pre_processing();
  gaussTaskSequential->run();
  gaussTaskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(output_data.size(), size);
}
