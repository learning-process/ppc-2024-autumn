#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shulpin_i_simpson_method/include/simpson_method.hpp"

constexpr double ESTIMATE = 1e-5;

TEST(shulpin_simpson_method_seq, pipeline_run) {
  double a = 0.0;
  double b = 1.0;
  double c = 0.0;
  double d = 1.0;

  int N = 4201;

  double global_integral = 0.0;
  double ref_integral = 0.38682;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  auto taskSequential = std::make_shared<shulpin_simpson_method::SimpsonMethodSeq>(taskDataSeq);
  taskSequential->set_seq(shulpin_simpson_method::f_sin_mul_cos);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_NEAR(global_integral, ref_integral, ESTIMATE);
}

TEST(shulpin_simpson_method_seq, task_run) {
  double a = 0.0;
  double b = 1.0;
  double c = 0.0;
  double d = 1.0;

  int N = 4201;

  double global_integral = 0.0;
  double ref_integral = 0.38682;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  auto taskSequential = std::make_shared<shulpin_simpson_method::SimpsonMethodSeq>(taskDataSeq);
  taskSequential->set_seq(shulpin_simpson_method::f_sin_mul_cos);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_NEAR(global_integral, ref_integral, ESTIMATE);
}