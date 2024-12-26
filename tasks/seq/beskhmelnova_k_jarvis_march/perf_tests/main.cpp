#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/beskhmelnova_k_jarvis_march/include/jarvis_march.hpp"

TEST(beskhmelnova_k_jarvis_march_seq, test_pipeline_run) {
  int num_points = 1000;
  std::vector<double> x(num_points);
  std::vector<double> y(num_points);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 4; i < num_points; i++) {
    x[i] = std::rand() % 1000;
    y[i] = std::rand() % 1000;
  }
  x[0] = -1.0;
  y[0] = -1.0;

  x[1] = -1.0;
  y[1] = 1000.0;

  x[2] = 1000.0;
  y[2] = 1000.0;

  x[3] = 1000.0;
  y[3] = -1.0;

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 4;
  std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
  std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  auto testTaskSequential = std::make_shared<beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double>>(taskDataSeq);

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

  EXPECT_EQ(res_size, hull_size);
  for (int i = 0; i < res_size; i++) {
    EXPECT_EQ(res_x[i], hull_x[i]);
    EXPECT_EQ(res_y[i], hull_y[i]);
  }
}

TEST(beskhmelnova_k_jarvis_march_seq, test_task_run) {
  int num_points = 1000;
  std::vector<double> x(num_points);
  std::vector<double> y(num_points);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 4; i < num_points; i++) {
    x[i] = std::rand() % 1000;
    y[i] = std::rand() % 1000;
  }
  x[0] = -1.0;
  y[0] = -1.0;

  x[1] = -1.0;
  y[1] = 1000.0;

  x[2] = 1000.0;
  y[2] = 1000.0;

  x[3] = 1000.0;
  y[3] = -1.0;

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 4;
  std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
  std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  auto testTaskSequential = std::make_shared<beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double>>(taskDataSeq);

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

  EXPECT_EQ(res_size, hull_size);
  for (int i = 0; i < res_size; i++) {
    EXPECT_EQ(res_x[i], hull_x[i]);
    EXPECT_EQ(res_y[i], hull_y[i]);
  }
}
