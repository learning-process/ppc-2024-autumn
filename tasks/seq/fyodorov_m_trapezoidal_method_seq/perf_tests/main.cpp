// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/fyodorov_m_trapezoidal_method_seq/include/ops_seq.hpp"

namespace {

bool almost_equal(double a, double b, double epsilon = 1e-6) { return std::abs(a - b) < epsilon; }

//double test_func_1(const std::vector<double>& x) { return x[0] * x[0] + x[1] * x[1]; }

double test_func_2(const std::vector<double>& x) { return x[0]; }

double test_func_3(const std::vector<double>& x) { return x[0] + x[1] + x[2]; }

//double test_func_4(const std::vector<double>& x) { return std::sin(x[0]); }

//double test_func_5(const std::vector<double>& x) { return x[0] * x[0] * x[0]; }

//double test_func_6(const std::vector<double>& x) { return x[0] * x[1]; }

// double test_func_7(const std::vector<double>& x) {
//   (void)x;
//   return 1.0;
// }

//double test_func_8(const std::vector<double>& x) { return x[0] * x[0]; }

// f(x, y) = x + y/2
//double test_func_9(const std::vector<double>& x) { return x[0] + x[1] / 2.0; }

// f(x) = x/2
//double test_func_10(const std::vector<double>& x) { return x[0] / 2.0; }

// f(x, y) = x^2 * y
//double test_func_11(const std::vector<double>& x) { return x[0] * x[0] * x[1]; }

// f(x, y, z) = x*y*z
//double test_func_12(const std::vector<double>& x) { return x[0] * x[1] * x[2]; }

}  // namespace

TEST(sequential_example_perf_test, test_int_task_pipeline_run) {
  // Create data
  std::function<double(const std::vector<double>&)> func = test_func_3;
  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {1.0};
  std::vector<int> intervals = {100};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(intervals.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  auto testTaskSequential = std::make_shared<fyodorov_m_trapezoidal_method_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_TRUE(almost_equal(out[0], 0.5));
}

TEST(sequential_example_perf_test, test_int_task_task_run) {
  // Create data
  std::function<double(const std::vector<double>&)> func = test_func_2;
  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {1.0};
  std::vector<int> intervals = {100};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(intervals.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<fyodorov_m_trapezoidal_method_seq::TestTaskSequential>(taskDataSeq);

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
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(almost_equal(out[0], 0.5));
}

/*
TEST(sequential_example_perf_test, test_pipeline_run) {
  const int count = 100;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<fyodorov_m_trapezoidal_method_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(count, out[0]);
}

TEST(sequential_example_perf_test, test_task_run) {
  const int count = 100;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<fyodorov_m_trapezoidal_method_seq::TestTaskSequential>(taskDataSeq);

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
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(count, out[0]);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
*/