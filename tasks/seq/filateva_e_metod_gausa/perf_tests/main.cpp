// Filateva Elizaveta Metod Gausa
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/filateva_e_metod_gausa/include/ops_seq.hpp"

std::vector<double> gereratorSLU(std::vector<double>& matrix, std::vector<double>& vecB) {
  int min_z = -5;
  int max_z = 5;
  int size = vecB.size();
  std::vector<double> resh(size);
  for (int i = 0; i < size; i++) {
    resh[i] = rand() % (max_z - min_z + 1) + min_z;
  }
  for (int i = 0; i < size; i++) {
    double sum = 0;
    double sumB = 0;
    for (int j = 0; j < size; j++) {
      matrix[i * size + j] = rand() % (max_z - min_z + 1) + min_z;
      sum += abs(matrix[i * size + j]);
    }
    matrix[i * size + i] = sum;
    for (int j = 0; j < size; j++) {
      sumB += matrix[i * size + j] * resh[j];
    }
    vecB[i] = sumB;
  }
  return resh;
}

bool check(std::vector<double>& resh, std::vector<double>& tResh, double alfa) {
  for (long unsigned int i = 0; i < tResh.size(); i++) {
    if (abs(resh[i] - tResh[i]) > alfa) {
      return false;
    }
  }
  return true;
}

TEST(filateva_e_metod_gausa_seq, test_pipeline_run) {
  int size = 800;
  double alfa = 0.000000001;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer;
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  auto metodGausa = std::make_shared<filateva_e_metod_gausa_seq::MetodGausa>(taskData);

  ASSERT_EQ(metodGausa->validation(), true);
  metodGausa->pre_processing();
  metodGausa->run();
  metodGausa->post_processing();
  auto *temp = reinterpret_cast<double*>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodGausa);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(check(answer, tResh, alfa), true);
}

TEST(filateva_e_metod_gausa_seq, test_task_run) {
  int size = 800;
  double alfa = 0.000000001;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer;
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  auto metodGausa = std::make_shared<filateva_e_metod_gausa_seq::MetodGausa>(taskData);

  ASSERT_EQ(metodGausa->validation(), true);
  metodGausa->pre_processing();
  metodGausa->run();
  metodGausa->post_processing();
  auto *temp = reinterpret_cast<double*>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodGausa);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(check(answer, tResh, alfa), true);
}
