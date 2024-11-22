#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/chernykh_a_adjust_image_contrast/include/ops_seq.hpp"
#include "seq/chernykh_a_adjust_image_contrast/include/pixel.hpp"

TEST(chernykh_a_adjust_image_contrast_seq, test_pipeline_run) {
  auto input_size = 10'000'000;
  auto contrast_factor = 1.5f;
  auto input = chernykh_a_adjust_image_contrast_seq::hex_colors_to_pixels(std::vector<uint32_t>(input_size, 0x906030));
  auto want = chernykh_a_adjust_image_contrast_seq::hex_colors_to_pixels(std::vector<uint32_t>(input_size, 0x985008));
  auto output = std::vector<chernykh_a_adjust_image_contrast_seq::Pixel>(input_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<chernykh_a_adjust_image_contrast_seq::SequentialTask>(task_data, contrast_factor);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = std::chrono::high_resolution_clock::now();
  perf_attributes->current_timer = [&] {
    auto current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current - start).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  perf_analyzer->pipeline_run(perf_attributes, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);
  ASSERT_EQ(want, output);
}

TEST(chernykh_a_adjust_image_contrast_seq, test_task_run) {
  auto input_size = 10'000'000;
  auto contrast_factor = 1.5f;
  auto input = chernykh_a_adjust_image_contrast_seq::hex_colors_to_pixels(std::vector<uint32_t>(input_size, 0x906030));
  auto want = chernykh_a_adjust_image_contrast_seq::hex_colors_to_pixels(std::vector<uint32_t>(input_size, 0x985008));
  auto output = std::vector<chernykh_a_adjust_image_contrast_seq::Pixel>(input_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<chernykh_a_adjust_image_contrast_seq::SequentialTask>(task_data, contrast_factor);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = std::chrono::high_resolution_clock::now();
  perf_attributes->current_timer = [&] {
    auto current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current - start).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  perf_analyzer->task_run(perf_attributes, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);
  ASSERT_EQ(want, output);
}
