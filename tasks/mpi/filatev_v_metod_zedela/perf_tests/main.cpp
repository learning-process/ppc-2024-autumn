// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/filatev_v_metod_zedela/include/ops_mpi.hpp"

namespace filatev_v_metod_zedela_mpi {

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

}  // namespace filatev_v_metod_zedela_mpi

TEST(filatev_v_metod_zedela_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = 500;
  double alfa = 0.0001;
  std::vector<double> answer;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix;
  std::vector<int> vecB;
  std::vector<int> resh;

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    filatev_v_metod_zedela_mpi::generatorMatrix(matrix, size);
    resh = filatev_v_metod_zedela_mpi::genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto metodZedela = std::make_shared<filatev_v_metod_zedela_mpi::MetodZedela>(taskData, world);
  metodZedela->setAlfa(alfa);

  ASSERT_EQ(metodZedela->validation(), true);
  metodZedela->pre_processing();
  metodZedela->run();
  metodZedela->post_processing();

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

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(filatev_v_metod_zedela_mpi::rightAns(answer, resh, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = 500;
  double alfa = 0.00001;
  std::vector<double> answer;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix;
  std::vector<int> vecB;
  std::vector<int> resh;

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    filatev_v_metod_zedela_mpi::generatorMatrix(matrix, size);
    resh = filatev_v_metod_zedela_mpi::genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto metodZedela = std::make_shared<filatev_v_metod_zedela_mpi::MetodZedela>(taskData, world);
  metodZedela->setAlfa(alfa);

  ASSERT_EQ(metodZedela->validation(), true);
  metodZedela->pre_processing();
  metodZedela->run();
  metodZedela->post_processing();

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
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);
    ASSERT_EQ(filatev_v_metod_zedela_mpi::rightAns(answer, resh, alfa), true);
  }
}
