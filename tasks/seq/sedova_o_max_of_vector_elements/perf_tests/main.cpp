// Copyright 2024 Sedova Olga
#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sedova_o_max_of_vector_elements/include/ops_seq.hpp"

std::vector<int> generate_random_vector(size_t size, size_t value) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = random() % (value + 1);
  }
  return vec;
}

std::vector<std::vector<int>> generate_random_matrix(size_t rows, size_t cols,
                                                                                          size_t value) {
  std::vector<std::vector<int>> matrix(rows);
  for (size_t i = 0; i < rows; i++) {
    matrix[i] = generate_random_vector(cols, value);
  }
  return matrix;
}

TEST(sedova_o_max_of_vector_elements_seq, test_pipeline_run_small_matrix) {
  std::random_device dev;
  std::mt19937 random(dev());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t size = 5000;
  size_t value = 5000;

  std::vector<std::vector<int>> in;
  in = sedova_o_max_of_vector_elements_seq::generate_random_matrix(size, size, value);
  std::vector<int32_t> out(1, in[0][0]);

  size_t rows = random() % size;
  size_t cols = random() % size;
  in[rows][cols] = value;

  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<sedova_o_max_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);

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
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(value, out[0]);
}

TEST(sedova_o_max_of_vector_elements_seq, test_pipeline_run_large_matrix) {
  std::random_device dev;
  std::mt19937 random(dev());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t size = 50000;
  size_t value = 50000;

  std::vector<std::vector<int>> in;
  in = sedova_o_max_of_vector_elements_seq::generate_random_matrix(size, size, value);
  std::vector<int32_t> out(1, in[0][0]);

  size_t rows = random() % size;
  size_t cols = random() % size;
  in[rows][cols] = value;

  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<sedova_o_max_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);

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
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(value, out[0]);
}

TEST(sedova_o_max_of_vector_elements_seq, test_pipeline_run_different_values) {
  std::random_device dev;
  std::mt19937 random(dev());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t size = 15000;
  size_t value = 15000;

  std::vector<std::vector<int>> in;
  in = sedova_o_max_of_vector_elements_seq::generate_random_matrix(size, size, value);
  std::vector<int32_t> out(1, in[0][0]);

  size_t rows = random() % size;
  size_t cols = random() % size;
  in[rows][cols] = value + 1;  // Изменил значение, чтобы найти максимальное

  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<sedova_o_max_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);

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
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(value + 1, out[0]);
}