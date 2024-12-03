// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kolokolova_d_gaussian_method_horizontal/include/ops_seq.hpp"

using namespace kolokolova_d_gaussian_method_horizontal_seq;

std::vector<int> kolokolova_d_gaussian_method_horizontal_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(-100, 100);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

TEST(kolokolova_d_gaussian_method_horizontal_seq, test_pipeline_run) {
  int count_equations = 250;

  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff(size_coef_mat, 0);
  std::vector<int> input_y(count_equations, 0);
  std::vector<double> func_res(count_equations, 0);

  input_y = kolokolova_d_gaussian_method_horizontal_seq::getRandomVector(count_equations);
  input_coeff = kolokolova_d_gaussian_method_horizontal_seq::getRandomVector(size_coef_mat);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
  taskDataSeq->inputs_count.emplace_back(input_coeff.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
  taskDataSeq->inputs_count.emplace_back(input_y.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  auto testTaskSequential =
      std::make_shared<kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential>(taskDataSeq);

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
  EXPECT_EQ(int(func_res.size()), count_equations);
}

TEST(kolokolova_d_gaussian_method_horizontal_seq, test_task_run) {
  int count_equations = 300;

  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff(size_coef_mat, 0);
  std::vector<int> input_y(count_equations, 0);
  std::vector<double> func_res(count_equations, 0);

  input_y = kolokolova_d_gaussian_method_horizontal_seq::getRandomVector(count_equations);
  input_coeff = kolokolova_d_gaussian_method_horizontal_seq::getRandomVector(size_coef_mat);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
  taskDataSeq->inputs_count.emplace_back(input_coeff.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
  taskDataSeq->inputs_count.emplace_back(input_y.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  auto testTaskSequential =
      std::make_shared<kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential>(taskDataSeq);

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

  EXPECT_EQ(int(func_res.size()), count_equations);
}