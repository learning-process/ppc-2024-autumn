#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/polikanov_v_gauss_band_columns/include/ops_seq.hpp"

namespace polikanov_v_gauss_band_columns_seq {

void generateGaussianCompatibleMatrix(int size, std::vector<double>& extendedMatrix, std::vector<double>& solutions) {
  solutions.resize(size);
  extendedMatrix.resize(size * (size + 1));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-100.0, 100.0);

  for (int i = 0; i < size; ++i) {
    solutions[i] = dist(gen);
  }

  for (int i = 0; i < size; ++i) {
    double sum = 0.0;
    for (int j = 0; j < size; ++j) {
      double value = dist(gen);
      extendedMatrix[i * (size + 1) + j] = value;
      sum += value * solutions[j];
    }
    extendedMatrix[i * (size + 1) + size] = sum;
  }
}

}  // namespace polikanov_v_gauss_band_columns_seq

TEST(polikanov_v_gauss_band_columns_seq, pipeline_run) {
  size_t n = 200;
  std::vector<double> input_matrix(n * (n + 1));
  std::vector<double> global_result(n);
  std::vector<double> exp_results(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  polikanov_v_gauss_band_columns_seq::generateGaussianCompatibleMatrix(n, input_matrix, exp_results);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  auto taskSequential = std::make_shared<polikanov_v_gauss_band_columns_seq::GaussBandColumnsSequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(exp_results.size(), global_result.size());
  for (size_t i = 0; i < exp_results.size(); i++) {
    EXPECT_NEAR(global_result[i], exp_results[i], 0.01);
  }
}

TEST(polikanov_v_gauss_band_columns_seq, task_run) {
  size_t n = 200;
  std::vector<double> input_matrix(n * (n + 1));
  std::vector<double> global_result(n);
  std::vector<double> exp_results(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  polikanov_v_gauss_band_columns_seq::generateGaussianCompatibleMatrix(n, input_matrix, exp_results);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  auto taskSequential = std::make_shared<polikanov_v_gauss_band_columns_seq::GaussBandColumnsSequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(exp_results.size(), global_result.size());
  for (size_t i = 0; i < exp_results.size(); i++) {
    EXPECT_NEAR(global_result[i], exp_results[i], 0.01);
  }
}
