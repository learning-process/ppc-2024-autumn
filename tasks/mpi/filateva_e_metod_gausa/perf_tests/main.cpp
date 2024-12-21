// Filateva Elizaveta Metod Gausa
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <limits>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/filateva_e_metod_gausa/include/ops_mpi.hpp"

#define alfa std::numeric_limits<double>::epsilon() * 1000000000

std::vector<double> gereratorSLU(std::vector<double> &matrix, std::vector<double> &vecB) {
  int min_z = -100;
  int max_z = 100;
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

TEST(filateva_e_metod_gausa_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = 800;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer;
  std::vector<double> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    answer.resize(size);
    tResh = gereratorSLU(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto metodGausa = std::make_shared<filateva_e_metod_gausa_mpi::MetodGausa>(taskData);
  ASSERT_TRUE(metodGausa->validation());
  metodGausa->pre_processing();
  metodGausa->run();
  metodGausa->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodGausa);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_NEAR(tResh[i], answer[i], alfa);
    }
  }
}

TEST(filateva_e_metod_gausa_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = 800;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer;
  std::vector<double> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    answer.resize(size);
    tResh = gereratorSLU(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  auto metodGausa = std::make_shared<filateva_e_metod_gausa_mpi::MetodGausa>(taskData);
  ASSERT_TRUE(metodGausa->validation());
  metodGausa->pre_processing();
  metodGausa->run();
  metodGausa->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodGausa);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_NEAR(tResh[i], answer[i], alfa);
    }
  }
}