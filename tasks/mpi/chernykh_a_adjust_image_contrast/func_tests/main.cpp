#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "mpi/chernykh_a_adjust_image_contrast/include/ops_mpi.hpp"
#include "mpi/chernykh_a_adjust_image_contrast/include/pixel.hpp"

namespace chernykh_a_adjust_image_contrast_mpi {

std::vector<uint32_t> random_hex_colors(size_t count) {
  auto dev = std::random_device();
  auto gen = std::mt19937(dev());
  auto dist = std::uniform_int_distribution<uint32_t>(0x000000, 0xFFFFFF);
  auto result = std::vector<uint32_t>(count);
  std::transform(result.begin(), result.end(), result.begin(), [&dist, &gen](uint32_t) { return dist(gen); });
  return result;
}

void run_valid_task(float contrast_factor, size_t output_size, const std::vector<uint32_t> &input_hex_colors) {
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

  auto par_task = ParallelTask(par_task_data, contrast_factor);
  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    auto seq_output = std::vector<Pixel>(output_size);
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    auto seq_task = SequentialTask(seq_task_data, contrast_factor);
    ASSERT_TRUE(seq_task.validation());
    ASSERT_TRUE(seq_task.pre_processing());
    ASSERT_TRUE(seq_task.run());
    ASSERT_TRUE(seq_task.post_processing());
    ASSERT_EQ(seq_output, par_output);
  }
}

void run_invalid_task(float contrast_factor, size_t output_size, const std::vector<uint32_t> &input_hex_colors) {
  auto world = boost::mpi::communicator();
  if (world.rank() == 0) {
    auto input = std::vector<Pixel>();

    auto par_output = std::vector<Pixel>(output_size);
    auto par_task_data = std::make_shared<ppc::core::TaskData>();
    input = hex_colors_to_pixels(input_hex_colors);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());

    auto par_task = ParallelTask(par_task_data, contrast_factor);
    ASSERT_FALSE(par_task.validation());

    auto seq_output = std::vector<Pixel>(output_size);
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    auto seq_task = SequentialTask(seq_task_data, contrast_factor);
    ASSERT_FALSE(seq_task.validation());
  }
}

}  // namespace chernykh_a_adjust_image_contrast_mpi

TEST(chernykh_a_adjust_image_contrast_mpi, contrast_clamped_to_max_value) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(2.0, 1, {0xFFFFFF});
}

TEST(chernykh_a_adjust_image_contrast_mpi, increases_color_contrast) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(1.7, 1, {0xFF5733});
}

TEST(chernykh_a_adjust_image_contrast_mpi, reduces_color_contrast) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(0.4, 1, {0xB0C4DE});
}

TEST(chernykh_a_adjust_image_contrast_mpi, contrast_factor_equals_zero_turns_color_gray) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(0.0, 1, {0x2C3E50});
}

TEST(chernykh_a_adjust_image_contrast_mpi, gray_color_remains_unchanged_with_factor_greater_than_one) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(1.5, 1, {0x808080});
}

TEST(chernykh_a_adjust_image_contrast_mpi, gray_color_remains_unchanged_with_factor_less_than_one) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(0.5, 1, {0x808080});
}

TEST(chernykh_a_adjust_image_contrast_mpi, contrast_factor_equals_one_does_not_change_color) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(1.0, 1, {0x2A8A5F});
}

TEST(chernykh_a_adjust_image_contrast_mpi, random_input_image) {
  chernykh_a_adjust_image_contrast_mpi::run_valid_task(
      1.4, 100'000, chernykh_a_adjust_image_contrast_mpi::random_hex_colors(100'000));
}

TEST(chernykh_a_adjust_image_contrast_mpi, negative_contrast_factor_fails_validation) {
  chernykh_a_adjust_image_contrast_mpi::run_invalid_task(-0.1, 1, {0x000000});
}

TEST(chernykh_a_adjust_image_contrast_mpi, contrast_factor_greater_than_two_fails_validation) {
  chernykh_a_adjust_image_contrast_mpi::run_invalid_task(2.1, 1, {0x000000});
}

TEST(chernykh_a_adjust_image_contrast_mpi, input_output_size_mismatch_fails_validation) {
  chernykh_a_adjust_image_contrast_mpi::run_invalid_task(1.0, 0, {0x000000});
}

TEST(chernykh_a_adjust_image_contrast_mpi, empty_image_fails_validation) {
  chernykh_a_adjust_image_contrast_mpi::run_invalid_task(0.0, 0, {});
}
