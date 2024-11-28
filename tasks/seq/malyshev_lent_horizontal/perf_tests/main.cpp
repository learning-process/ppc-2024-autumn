// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/malyshev_lent_horizontal/include/ops_seq.hpp"

namespace malyshev_lent_horizontal {

std::vector<std::vector<int32_t>> generateRandomMatrix(uint32_t rows, uint32_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> data(rows, std::vector<int32_t>(cols));

  for (auto &row : data) {
    for (auto &el : row) {
      el = -200 + gen() % (300 + 200 + 1);
    }
  }

  return data;
}

std::vector<int32_t> generateRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> data(size);

  for (auto &el : data) {
    el = -200 + gen() % (300 + 200 + 1);
  }

  return data;
}

}  // namespace malyshev_lent_horizontal

TEST(malyshev_lent_horizontal, test_pipeline_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> seqResult;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
  randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
  seqResult.resize(rows);

  for (auto &row : randomMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(seqResult.size());

  auto testSeqTask = std::make_shared<malyshev_lent_horizontal::TestTaskSequential>(taskDataSeq);

  ASSERT_TRUE(testSeqTask->validation());
  ASSERT_TRUE(testSeqTask->pre_processing());
  ASSERT_TRUE(testSeqTask->run());
  ASSERT_TRUE(testSeqTask->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start_time).count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(malyshev_lent_horizontal, test_task_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> seqResult;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
  randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
  seqResult.resize(rows);

  for (auto &row : randomMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(seqResult.size());

  auto testSeqTask = std::make_shared<malyshev_lent_horizontal::TestTaskSequential>(taskDataSeq);

  ASSERT_TRUE(testSeqTask->validation());
  ASSERT_TRUE(testSeqTask->pre_processing());
  ASSERT_TRUE(testSeqTask->run());
  ASSERT_TRUE(testSeqTask->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start_time).count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}