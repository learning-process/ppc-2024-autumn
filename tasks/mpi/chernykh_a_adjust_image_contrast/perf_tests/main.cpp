#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chernykh_a_adjust_image_contrast/include/ops_mpi.hpp"
#include "mpi/chernykh_a_adjust_image_contrast/include/pixel.hpp"

namespace chernykh_a_adjust_image_contrast_mpi {

enum class RunType : uint8_t { TASK, PIPELINE };

bool all_pixels_match(const std::vector<Pixel> &pixels, Pixel expected_pixel) {
  return std::all_of(pixels.begin(), pixels.end(),
                     [&expected_pixel](const Pixel &pixel) { return pixel == expected_pixel; });
}

void run_task(RunType run_type, float contrast_factor, size_t output_size,
              const std::vector<uint32_t> &input_hex_colors, uint32_t want_hex_color) {
  auto world = boost::mpi::communicator();
  auto input = std::vector<Pixel>();

  auto par_output = std::vector<Pixel>(output_size);
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = hex_colors_to_pixels(input_hex_colors);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  auto par_task = std::make_shared<ParallelTask>(par_task_data, contrast_factor);

  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = boost::mpi::timer();
  perf_attributes->current_timer = [&] { return start.elapsed(); };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(par_task);

  switch (run_type) {
    case RunType::PIPELINE:
      perf_analyzer->pipeline_run(perf_attributes, perf_results);
      break;
    case RunType::TASK:
      perf_analyzer->task_run(perf_attributes, perf_results);
      break;
  }

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    EXPECT_TRUE(all_pixels_match(par_output, Pixel::from_hex_color(want_hex_color)));
  }
}

}  // namespace chernykh_a_adjust_image_contrast_mpi

TEST(chernykh_a_adjust_image_contrast_mpi, test_pipeline_run) {
  chernykh_a_adjust_image_contrast_mpi::run_task(chernykh_a_adjust_image_contrast_mpi::RunType::PIPELINE, 1.5f,
                                                 1'000'000, std::vector<uint32_t>(1'000'000, 0x906030), 0x985008);
}

TEST(chernykh_a_adjust_image_contrast_mpi, test_task_run) {
  chernykh_a_adjust_image_contrast_mpi::run_task(chernykh_a_adjust_image_contrast_mpi::RunType::TASK, 1.5f, 1'000'000,
                                                 std::vector<uint32_t>(1'000'000, 0x906030), 0x985008);
}
