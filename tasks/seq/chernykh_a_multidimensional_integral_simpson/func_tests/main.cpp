#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <numbers>

#include "seq/chernykh_a_multidimensional_integral_simpson/include/ops_seq.hpp"

namespace chernykh_a_multidimensional_integral_simpson_seq {

void run_valid_task(func_nd_t func, bounds_t& bounds, steps_t& steps, double want, double tolerance) {
  auto output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.emplace_back(bounds.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data->inputs_count.emplace_back(steps.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = SequentialTask(task_data, func);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  EXPECT_NEAR(want, output, tolerance);
}

void run_invalid_task(func_nd_t func, bounds_t& bounds, steps_t& steps) {
  auto output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.emplace_back(bounds.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data->inputs_count.emplace_back(steps.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = SequentialTask(task_data, func);
  ASSERT_FALSE(task.validation());
}

}  // namespace chernykh_a_multidimensional_integral_simpson_seq

namespace chernykh_a_mis_seq = chernykh_a_multidimensional_integral_simpson_seq;

TEST(chernykh_a_multidimensional_integral_simpson_seq, linear_2d_integration) {
  auto func = [](const auto& args) { return (2 * args[0]) + (3 * args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 1.0}, {0.0, 2.0}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  auto want = 8.0;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, quadratic_3d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]) + (args[2] * args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2, 2};
  auto want = 1.0;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, exponential_3d_integration) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1] + args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}};
  auto steps = chernykh_a_mis_seq::steps_t{4, 4, 4};
  auto want = std::pow(std::exp(0.5) - 1, 3);
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, trigonometric_2d_integration) {
  auto func = [](const auto& args) { return std::sin(args[0]) * std::cos(args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}};
  auto steps = chernykh_a_mis_seq::steps_t{20, 20};
  auto want = 2.0;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, polynomial_3d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[1]) + (args[1] * args[2]) + (args[0] * args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2, 2};
  auto want = 0.75;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, linear_3d_integration) {
  auto func = [](const auto& args) { return args[0] + (2 * args[1]) + (3 * args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 2.0}, {0.0, 1.0}, {0.0, 3.0}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2, 2};
  auto want = 39.0;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, quadratic_2d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 2.0}, {0.0, 3.0}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  auto want = 26.0;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, exponential_2d_integration) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_seq::steps_t{10, 10};
  auto want = std::pow((std::numbers::e - 1), 2);
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, trigonometric_3d_integration) {
  auto func = [](const auto& args) { return std::sin(args[0]) * std::cos(args[1]) * std::tan(args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{
      {0.0, std::numbers::pi},
      {0.0, std::numbers::pi / 2},
      {0.0, std::numbers::pi / 4},
  };
  auto steps = chernykh_a_mis_seq::steps_t{16, 16, 16};
  auto want = std::numbers::ln2;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, cubic_2d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0] * args[0]) + (args[1] * args[1] * args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 1.0}, {0.0, 2.0}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  auto want = 4.5;
  auto tolerance = 1e-5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, small_range_linear_2d_integration) {
  auto func = [](const auto& args) { return (2 * args[0]) + (3 * args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.00000001, 0.00000003}, {0.00000002, 0.00000004}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  auto want = 5.2e-23;
  auto tolerance = 1e-23;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, small_range_quadratic_2d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.00000005, 0.00000006}, {0.00000001, 0.00000002}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  auto want = 3.26666666666667e-31;
  auto tolerance = 1e-32;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, small_range_exponential_2d_integration) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.00000003, 0.00000005}, {0.00000002, 0.00000004}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  auto want = 4.00000029906011e-16;
  auto tolerance = 1e-16;
  chernykh_a_mis_seq::run_valid_task(func, bounds, steps, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, empty_bounds_fails_validation) {
  auto func = [](const auto& args) { return (2 * args[0]) + (3 * args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  chernykh_a_mis_seq::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, incorrect_bounds_fails_validation) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1] + args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 0.5}, {1.0, 0.5}, {0.0, 0.5}};
  auto steps = chernykh_a_mis_seq::steps_t{4, 4, 4};
  chernykh_a_mis_seq::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, empty_steps_fails_validation) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]) + (args[2] * args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_seq::steps_t{};
  chernykh_a_mis_seq::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, incorrect_steps_fails_validation) {
  auto func = [](const auto& args) { return std::sin(args[0]) * std::cos(args[1]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}};
  auto steps = chernykh_a_mis_seq::steps_t{20, 21};
  chernykh_a_mis_seq::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, bounds_steps_size_mismatch_fails_validation) {
  auto func = [](const auto& args) { return (args[0] * args[1]) + (args[1] * args[2]) + (args[0] * args[2]); };
  auto bounds = chernykh_a_mis_seq::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_seq::steps_t{2, 2};
  chernykh_a_mis_seq::run_invalid_task(func, bounds, steps);
}
