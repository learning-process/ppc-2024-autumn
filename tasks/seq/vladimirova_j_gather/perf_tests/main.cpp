// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vladimirova_j_gather/include/ops_seq.hpp"
using namespace std::chrono_literals;

namespace vladimirova_j_gather_seq {
std::vector<int> getRandomVal(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  vec[0] = 2;
  vec[sz - 1] = 2;
  for (int i = 1; i < sz - 1; i++) {
    if ((i != 0) && (vec[i - 1] != 2)) {
      vec[i] = 2;
      continue;
    }
    vec[i] = (gen() % 3 - 1);
    if (vec[i] == 0) vec[i] = 2;
  }
  return vec;
};
}  // namespace vladimirova_j_gather_seq

TEST(sequential_vladimirova_j_gather, test_pipeline_run) {
  // Create data
  std::vector<int> global_vector;
  std::vector<int> out(1, 0);

  int d_end_count = 500;
  int noDEnd = 0;
  for (int j = 0; j < d_end_count; j++) {
    std::vector<int> some_dead_end;
    std::vector<int> tmp;
    some_dead_end = vladimirova_j_gather_seq::getRandomVal(15);
    tmp = vladimirova_j_gather_seq::getRandomVal(15);
    noDEnd += 15;
    global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
    global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
    global_vector.push_back(-1);
    global_vector.push_back(-1);
    noDEnd += 2;
    for (int i = some_dead_end.size() - 1; i >= 0; i--) {
      if (some_dead_end[i] != 2) {
        global_vector.push_back(-1 * some_dead_end[i]);
      } else {
        global_vector.push_back(2);
      }
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  taskDataSeq->outputs_count.emplace_back(ans_buf_vec.size());

  // Create Task
  auto testTaskSequential = std::make_shared<vladimirova_j_gather_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
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
  ASSERT_EQ((noDEnd >= (int)taskDataSeq->outputs_count[0]), 1);
}

TEST(sequential_vladimirova_j_gather, test_task_run) {
  // Create data
  std::vector<int> global_vector;
  std::vector<int> out(1, 0);

  int d_end_count = 900;
  int noDEnd = 0;
  for (int j = 0; j < d_end_count; j++) {
    std::vector<int> some_dead_end;
    std::vector<int> tmp;
    some_dead_end = vladimirova_j_gather_seq::getRandomVal(15);
    tmp = vladimirova_j_gather_seq::getRandomVal(15);
    noDEnd += 15;
    global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
    global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
    global_vector.push_back(-1);
    global_vector.push_back(-1);
    noDEnd += 2;
    for (int i = some_dead_end.size() - 1; i >= 0; i--) {
      if (some_dead_end[i] != 2) {
        global_vector.push_back(-1 * some_dead_end[i]);
      } else {
        global_vector.push_back(2);
      }
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  taskDataSeq->outputs_count.emplace_back(ans_buf_vec.size());

  // Create Task
  auto testTaskSequential = std::make_shared<vladimirova_j_gather_seq::TestTaskSequential>(taskDataSeq);

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
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ((noDEnd >= (int)taskDataSeq->outputs_count[0]), 1);
}