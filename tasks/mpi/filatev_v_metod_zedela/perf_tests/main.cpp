// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/filatev_v_metod_zedela/include/ops_mpi.hpp"

TEST(filatev_v_metod_zedela_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = 500;
  double alfa = 0.0001;
  std::vector<double> answer;
  filatev_v_metod_zedela_mpi::TestClassForMetodZedela test;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix;
  std::vector<int> vecB;

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    test.generatorMatrix(matrix, size);
    test.genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto metodZedela = std::make_shared<filatev_v_metod_zedela_mpi::MetodZedela>(taskData);
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

    ASSERT_EQ(test.rightAns(answer, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = 500;
  double alfa = 0.00001;
  std::vector<double> answer;
  filatev_v_metod_zedela_mpi::TestClassForMetodZedela test;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix;
  std::vector<int> vecB;

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    test.generatorMatrix(matrix, size);
    test.genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto metodZedela = std::make_shared<filatev_v_metod_zedela_mpi::MetodZedela>(taskData);
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
    ASSERT_EQ(test.rightAns(answer, alfa), true);
  }
}
