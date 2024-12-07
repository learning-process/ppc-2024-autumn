// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/filatev_v_metod_zedela/include/ops_seq.hpp"

namespace filatev_v_metod_zedela_seq {

int generatorVector(std::vector<int> &vec) {
  int sum = 0;
  for (long unsigned int i = 0; i < vec.size(); ++i) {
    vec[i] = rand() % 100 - 50;
    sum += abs(vec[i]);
  }
  return sum;
}

void generatorMatrix(std::vector<int> &matrix, int size) {
  for (int i = 0; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < size; ++j) {
      matrix[i * size + j] = rand() % 100 - 50;
      sum += abs(matrix[i * size + j]);
    }
    matrix[i * size + i] = sum + rand() % 100;
  }
}

std::vector<int> genetatirVectorB(std::vector<int> &matrix, std::vector<int> &vecB) {
  int size = vecB.size();
  std::vector<int> ans(size);
  generatorVector(ans);
  for (int i = 0; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < size; ++j) {
      sum += matrix[j + i * size] * ans[j];
    }
    vecB[i] = sum;
  }
  return ans;
}

bool rightAns(std::vector<double> &ans, std::vector<int> &resh, double alfa) {
  double max_r = 0;
  for (long unsigned int i = 0; i < ans.size(); ++i) {
    double temp = abs(ans[i] - resh[i]);
    max_r = std::max(max_r, temp);
  }
  return max_r < alfa;
}

}  // namespace filatev_v_metod_zedela_seq

TEST(filatev_v_metod_zedela_seq, test_pipeline_run) {
  int size = 500;
  double alfa = 0.00001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;
  std::vector<int> resh;

  filatev_v_metod_zedela_seq::generatorMatrix(matrix, size);
  resh = filatev_v_metod_zedela_seq::genetatirVectorB(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  auto metodZedela = std::make_shared<filatev_v_metod_zedela_seq::MetodZedela>(taskData);
  metodZedela->setAlfa(alfa);

  ASSERT_EQ(metodZedela->validation(), true);
  metodZedela->pre_processing();
  metodZedela->run();
  metodZedela->post_processing();
  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodZedela);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(filatev_v_metod_zedela_seq::rightAns(answer, resh, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_task_run) {
  int size = 500;
  double alfa = 0.00001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;
  std::vector<int> resh;

  filatev_v_metod_zedela_seq::generatorMatrix(matrix, size);
  resh = filatev_v_metod_zedela_seq::genetatirVectorB(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  auto metodZedela = std::make_shared<filatev_v_metod_zedela_seq::MetodZedela>(taskData);
  metodZedela->setAlfa(alfa);

  ASSERT_EQ(metodZedela->validation(), true);
  metodZedela->pre_processing();
  metodZedela->run();
  metodZedela->post_processing();
  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodZedela);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(filatev_v_metod_zedela_seq::rightAns(answer, resh, alfa), true);
}
