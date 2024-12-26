// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/koshkin_m_scalar_product_of_vectors/include/ops_seq.hpp"

static int offset = 0;

int koshkin_m_scalar_product_of_vectors::calculateDotProduct(const std::vector<int> &vec_1,
                                                             const std::vector<int> &vec_2) {
  long result = 0;
  for (size_t i = 0; i < vec_1.size(); i++) result += vec_1[i] * vec_2[i];
  return result;
}

std::vector<int> koshkin_m_scalar_product_of_vectors::generateRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) vec[i] = gen() % 100;
  return vec;
}

TEST(koshkin_m_scalar_product_of_vectors, test_pipeline_run) {
  const int count = 22800000;
  std::vector<int> out(1, 0);

  std::vector<int> vec_1 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);
  std::vector<int> vec_2 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_2.data()));

  taskDataSeq->inputs_count.emplace_back(vec_1.size());
  taskDataSeq->inputs_count.emplace_back(vec_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<koshkin_m_scalar_product_of_vectors::VectorDotProduct>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  int answer = koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2);
  ASSERT_EQ(answer, out[0]);
}

TEST(koshkin_m_scalar_product_of_vectors, test_task_run) {
  const int count = 22800000;
  std::vector<int> out(1, 0);

  std::vector<int> vec_1 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);
  std::vector<int> vec_2 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_2.data()));

  taskDataSeq->inputs_count.emplace_back(vec_1.size());
  taskDataSeq->inputs_count.emplace_back(vec_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<koshkin_m_scalar_product_of_vectors::VectorDotProduct>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  int answer = koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2);
  ASSERT_EQ(answer, out[0]);
}
