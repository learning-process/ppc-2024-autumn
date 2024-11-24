#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/chernykh_a_adjust_image_contrast/include/ops_seq.hpp"
#include "seq/chernykh_a_adjust_image_contrast/include/pixel.hpp"

namespace chernykh_a_adjust_image_contrast_seq {

enum class RunType : uint8_t { TASK, PIPELINE };

bool all_pixels_match(const std::vector<Pixel> &pixels, Pixel expected_pixel) {
  return std::all_of(pixels.begin(), pixels.end(),
                     [&expected_pixel](const Pixel &pixel) { return pixel == expected_pixel; });
}

void run_task(RunType run_type, float contrast_factor, size_t output_size,
              const std::vector<std::uint32_t> &input_hex_colors, std::uint32_t want_hex_color) {
  auto input = hex_colors_to_pixels(input_hex_colors);
  auto output = std::vector<Pixel>(output_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<SequentialTask>(task_data, contrast_factor);

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

  switch (run_type) {
    case RunType::PIPELINE:
      perf_analyzer->pipeline_run(perf_attributes, perf_results);
      break;
    case RunType::TASK:
      perf_analyzer->task_run(perf_attributes, perf_results);
      break;
  }

  ppc::core::Perf::print_perf_statistic(perf_results);
  ASSERT_TRUE(all_pixels_match(output, Pixel::from_hex_color(want_hex_color)));
}

}  // namespace chernykh_a_adjust_image_contrast_seq

TEST(chernykh_a_adjust_image_contrast_seq, test_pipeline_run) {
  chernykh_a_adjust_image_contrast_seq::run_task(chernykh_a_adjust_image_contrast_seq::RunType::PIPELINE, 1.5,
                                                 10'000'000, std::vector<std::uint32_t>(10'000'000, 0x906030),
                                                 0x985008);
}

TEST(chernykh_a_adjust_image_contrast_seq, test_task_run) {
  chernykh_a_adjust_image_contrast_seq::run_task(chernykh_a_adjust_image_contrast_seq::RunType::TASK, 1.5, 10'000'000,
                                                 std::vector<std::uint32_t>(10'000'000, 0x906030), 0x985008);
}
