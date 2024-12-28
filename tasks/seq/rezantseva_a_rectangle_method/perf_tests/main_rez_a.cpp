// PERF_TEST_SEQ_RECTANGLE
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/rezantseva_a_rectangle_method/include/ops_seq_rez_a.hpp"

TEST(rezantseva_a_rectangle_method_seq, test_pipeline_run) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0] - 2 * x[1] + 8 * x[2];
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {-6, 10};
  bounds[1] = {5, 27};
  bounds[2] = {15, 32};
  distrib[0] = 500;
  distrib[1] = 250;
  distrib[2] = 150;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  auto* bounds_data = new std::vector<std::pair<double, double>>(bounds);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds_data));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  auto* distrib_data = new std::vector<int>(distrib);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(distrib_data));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto rectangleMethodSequential =
      std::make_shared<rezantseva_a_rectangle_method_seq::RectangleMethodSequential>(taskDataSeq, function);

  ASSERT_EQ(rectangleMethodSequential->validation(), true);
  rectangleMethodSequential->pre_processing();
  rectangleMethodSequential->run();
  rectangleMethodSequential->post_processing();

  // Create Perf
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(rectangleMethodSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  double error = 0.01;
  ASSERT_NEAR(1085098.16, out[0], error);
  delete bounds_data;
  delete distrib_data;
}

TEST(rezantseva_a_rectangle_method_seq, test_task_run) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0] - 2 * x[1] + 8 * x[2];
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {-6, 10};
  bounds[1] = {5, 27};
  bounds[2] = {15, 32};
  distrib[0] = 500;
  distrib[1] = 250;
  distrib[2] = 150;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  auto* bounds_data = new std::vector<std::pair<double, double>>(bounds);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds_data));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  auto* distrib_data = new std::vector<int>(distrib);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(distrib_data));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto rectangleMethodSequential =
      std::make_shared<rezantseva_a_rectangle_method_seq::RectangleMethodSequential>(taskDataSeq, function);

  ASSERT_EQ(rectangleMethodSequential->validation(), true);
  rectangleMethodSequential->pre_processing();
  rectangleMethodSequential->run();
  rectangleMethodSequential->post_processing();

  // Create Perf
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(rectangleMethodSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  double error = 0.01;
  ASSERT_NEAR(1085098.16, out[0], error);
  delete bounds_data;
  delete distrib_data;
}
