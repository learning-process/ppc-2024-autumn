#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "seq/chernykh_a_adjust_image_contrast/include/ops_seq.hpp"
#include "seq/chernykh_a_adjust_image_contrast/include/pixel.hpp"

namespace chernykh_a_adjust_image_contrast_seq {

void run_valid_task(float contrast_factor, size_t output_size, const std::vector<std::uint32_t>& input_hex_colors,
                    const std::vector<std::uint32_t>& want_hex_colors) {
  auto input = hex_colors_to_pixels(input_hex_colors);
  auto want = hex_colors_to_pixels(want_hex_colors);
  auto output = std::vector<Pixel>(output_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = SequentialTask(task_data, contrast_factor);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  ASSERT_EQ(want, output);
}

void run_invalid_task(float contrast_factor, size_t output_size, const std::vector<std::uint32_t>& input_hex_colors) {
  auto input = hex_colors_to_pixels(input_hex_colors);
  auto output = std::vector<Pixel>(output_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = SequentialTask(task_data, contrast_factor);
  ASSERT_FALSE(task.validation());
}

}  // namespace chernykh_a_adjust_image_contrast_seq

TEST(chernykh_a_adjust_image_contrast_seq, contrast_clamped_to_max_value) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(2.0, 1, {0xFFFFFF}, {0xFFFFFF});
}

TEST(chernykh_a_adjust_image_contrast_seq, increases_color_contrast) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(1.7, 1, {0xFF5733}, {0xFF3A00});
}

TEST(chernykh_a_adjust_image_contrast_seq, reduces_color_contrast) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(0.4, 1, {0xB0C4DE}, {0x939BA6});
}

TEST(chernykh_a_adjust_image_contrast_seq, contrast_factor_equals_zero_turns_color_gray) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(0.0, 1, {0x2C3E50}, {0x808080});
}

TEST(chernykh_a_adjust_image_contrast_seq, gray_color_remains_unchanged_with_factor_greater_than_one) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(1.5, 1, {0x808080}, {0x808080});
}

TEST(chernykh_a_adjust_image_contrast_seq, gray_color_remains_unchanged_with_factor_less_than_one) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(0.5, 1, {0x808080}, {0x808080});
}

TEST(chernykh_a_adjust_image_contrast_seq, contrast_factor_equals_one_does_not_change_color) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(1.0, 1, {0x2A8A5F}, {0x2A8A5F});
}

TEST(chernykh_a_adjust_image_contrast_seq, large_input_image) {
  chernykh_a_adjust_image_contrast_seq::run_valid_task(1.5, 100'000, std::vector<std::uint32_t>(100'000, 0x906030),
                                                       std::vector<std::uint32_t>(100'000, 0x985008));
}

TEST(chernykh_a_adjust_image_contrast_seq, negative_contrast_factor_fails_validation) {
  chernykh_a_adjust_image_contrast_seq::run_invalid_task(-0.1, 1, {0x000000});
}

TEST(chernykh_a_adjust_image_contrast_seq, contrast_factor_greater_than_two_fails_validation) {
  chernykh_a_adjust_image_contrast_seq::run_invalid_task(2.1, 1, {0x000000});
}

TEST(chernykh_a_adjust_image_contrast_seq, input_output_size_mismatch_fails_validation) {
  chernykh_a_adjust_image_contrast_seq::run_invalid_task(1.0, 0, {0x000000});
}

TEST(chernykh_a_adjust_image_contrast_seq, empty_image_fails_validation) {
  chernykh_a_adjust_image_contrast_seq::run_invalid_task(0.0, 0, {});
}
