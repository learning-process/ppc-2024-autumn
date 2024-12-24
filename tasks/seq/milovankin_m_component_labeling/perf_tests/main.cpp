#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/milovankin_m_component_labeling/include/component_labeling_seq.hpp"

static std::vector<uint8_t> make_random_bin_img(size_t rows, size_t cols) {
  std::vector<uint8_t> img(rows * cols);
  for (auto& px : img) px = rand() % 2;

  return img;
}

TEST(milovankin_m_component_labeling_seq, test_pipeline_run) {
  size_t rows = 1000;
  size_t cols = 1000;
  std::vector<uint8_t> image = make_random_bin_img(rows, cols);
  std::vector<uint32_t> output(rows * cols);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(cols);

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataPar->outputs_count.emplace_back(rows);
  taskDataPar->outputs_count.emplace_back(cols);

  auto componentLabeling = std::make_shared<milovankin_m_component_labeling_seq::ComponentLabelingSeq>(taskDataPar);

  ASSERT_TRUE(componentLabeling->validation());
  ASSERT_TRUE(componentLabeling->pre_processing());
  componentLabeling->run();
  componentLabeling->post_processing();

  // Set up Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer and run
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(componentLabeling);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Not asserting results, as there's no way to get the expected result analytically
  // and class functionality is already tested in func_tests anyway
}

TEST(milovankin_m_component_labeling_seq, test_task_run) {
  size_t rows = 1000;
  size_t cols = 1000;
  std::vector<uint8_t> image = make_random_bin_img(rows, cols);
  std::vector<uint32_t> output(rows * cols);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(cols);

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataPar->outputs_count.emplace_back(rows);
  taskDataPar->outputs_count.emplace_back(cols);

  auto componentLabeling = std::make_shared<milovankin_m_component_labeling_seq::ComponentLabelingSeq>(taskDataPar);

  ASSERT_TRUE(componentLabeling->validation());
  ASSERT_TRUE(componentLabeling->pre_processing());
  componentLabeling->run();
  componentLabeling->post_processing();

  // Set up Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer and run
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(componentLabeling);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Not asserting results, as there's no way to get the expected result analytically
  // and class functionality is already tested in func_tests anyway
}
