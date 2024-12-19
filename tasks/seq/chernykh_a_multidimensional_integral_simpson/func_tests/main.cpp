#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <numbers>

#include "seq/chernykh_a_multidimensional_integral_simpson/include/ops_seq.hpp"

namespace chernykh_a_multidimensional_integral_simpson_seq {

void run_valid_task(func_nd_t func, bounds_t& bounds, step_range_t& step_range, double tolerance, double want) {
  auto output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.emplace_back(bounds.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_range));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = SequentialTask(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  EXPECT_NEAR(want, output, tolerance);
}

void run_invalid_task(func_nd_t func, bounds_t& bounds, step_range_t& step_range, double tolerance) {
  auto output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.emplace_back(bounds.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_range));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = SequentialTask(task_data);
  ASSERT_FALSE(task.validation());
}

}  // namespace chernykh_a_multidimensional_integral_simpson_seq

namespace chernykh_a_mis_seq = chernykh_a_multidimensional_integral_simpson_seq;

TEST(chernykh_a_multidimensional_integral_simpson_seq, linear_2d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double { return (2 * args[0]) + (3 * args[1]); };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 1.0}, {0.0, 2.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = 8.0;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, quadratic_3d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return (args[0] * args[0]) + (args[1] * args[1]) + (args[2] * args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = 1.0;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, exponential_3d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return std::exp(args[0] + args[1] + args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = std::pow(std::exp(0.5) - 1, 3);
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, trigonometric_2d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return std::sin(args[0]) * std::cos(args[1]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = 2.0;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, polynomial_3d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return (args[0] * args[1]) + (args[1] * args[2]) + (args[0] * args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = 0.75;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, linear_3d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return args[0] + (2 * args[1]) + (3 * args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 2.0}, {0.0, 1.0}, {0.0, 3.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = 39.0;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, quadratic_2d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return (args[0] * args[0]) + (args[1] * args[1]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 2.0}, {0.0, 3.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = 26.0;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, exponential_2d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double { return std::exp(args[0] + args[1]); };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 1.0}, {0.0, 1.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = (std::numbers::e - 1) * (std::numbers::e - 1);
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, trigonometric_3d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return std::sin(args[0]) * std::cos(args[1]) * std::tan(args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {
      {0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}, {0.0, std::numbers::pi / 4}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = std::numbers::ln2;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, cubic_2d_integration) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return (args[0] * args[0] * args[0]) + (args[1] * args[1] * args[1]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 1.0}, {0.0, 2.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  double want = 4.5;
  chernykh_a_mis_seq::run_valid_task(func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, empty_bounds_fails_validation) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return args[0] + (2 * args[1]) + (3 * args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  chernykh_a_mis_seq::run_invalid_task(func, bounds, step_range, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, incorrect_bounds_fails_validation) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return (args[0] * args[0]) + (args[1] * args[1]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 2.0}, {4.0, 3.0}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 1e-5;
  chernykh_a_mis_seq::run_invalid_task(func, bounds, step_range, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, incorrect_step_range_fails_validation) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return std::sin(args[0]) * std::cos(args[1]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}};
  chernykh_a_mis_seq::step_range_t step_range = {4, 2};
  double tolerance = 1e-5;
  chernykh_a_mis_seq::run_invalid_task(func, bounds, step_range, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, min_step_not_multiple_of_two_fails_validation) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return (args[0] * args[1]) + (args[1] * args[2]) + (args[0] * args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  chernykh_a_mis_seq::step_range_t step_range = {5, 100};
  double tolerance = 1e-5;
  chernykh_a_mis_seq::run_invalid_task(func, bounds, step_range, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, min_step_less_than_two_fails_validation) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double { return std::exp(args[0] + args[1]); };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 1.0}, {0.0, 1.0}};
  chernykh_a_mis_seq::step_range_t step_range = {0, 100};
  double tolerance = 1e-5;
  chernykh_a_mis_seq::run_invalid_task(func, bounds, step_range, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, non_positive_tolerance_fails_validation) {
  auto func = [](const chernykh_a_mis_seq::func_args_t& args) -> double {
    return std::exp(args[0] + args[1] + args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 100};
  double tolerance = 0;
  chernykh_a_mis_seq::run_invalid_task(func, bounds, step_range, tolerance);
}
