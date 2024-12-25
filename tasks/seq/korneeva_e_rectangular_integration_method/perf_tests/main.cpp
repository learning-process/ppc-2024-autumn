#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/korneeva_e_rectangular_integration_method/include/ops_seq.hpp"

TEST(korneeva_e_rectangular_integration_method_seq, test_pipeline_run) {
  // Integration limits and the integration function
  std::vector<std::pair<double, double>> limits(3, {-1000000, 1000000});

  // The function to integrate
  auto func = [](const std::vector<double> &args) -> double {
    return args.at(0) + args.at(1) + args.at(2);  // Example function f(x, y, z) = x + y + z
  };

  double out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  double eps = 1e-4;  // Accuracy
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create the task
  auto testTaskSequential =
      std::make_shared<korneeva_e_rectangular_integration_method_seq::RectangularIntegration>(taskDataSeq, func);

  // Create performance attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create and run the performance analysis
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Output performance statistics
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(korneeva_e_rectangular_integration_method_seq, test_task_run) {
  // Integration limits and the integration function
  std::vector<std::pair<double, double>> limits(3, {-1000000, 1000000});

  // The function to integrate
  auto func = [](const std::vector<double> &args) -> double {
    return args.at(0) + args.at(1) + args.at(2);  // Example function f(x, y, z) = x + y + z
  };

  double out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  double eps = 1e-4;  // Accuracy
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create the task
  auto testTaskSequential =
      std::make_shared<korneeva_e_rectangular_integration_method_seq::RectangularIntegration>(taskDataSeq, func);

  // Create performance attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create and run the performance analysis
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  // Output performance statistics
  ppc::core::Perf::print_perf_statistic(perfResults);
}
