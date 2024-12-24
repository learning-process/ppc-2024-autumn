#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>

#include "mpi/chernykh_a_multidimensional_integral_simpson/include/ops_mpi.hpp"

namespace chernykh_a_multidimensional_integral_simpson_mpi {

void run_valid_task(func_nd_t func, bounds_t& bounds, steps_t& steps, double tolerance) {
  auto world = boost::mpi::communicator();

  auto par_output = 0.0;
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds.data()));
    par_task_data->inputs_count.emplace_back(bounds.size());
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    par_task_data->inputs_count.emplace_back(steps.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&par_output));
    par_task_data->outputs_count.emplace_back(1);
  }

  auto par_task = ParallelTask(par_task_data, func);
  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    auto seq_output = 0.0;
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds.data()));
    seq_task_data->inputs_count.emplace_back(bounds.size());
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    seq_task_data->inputs_count.emplace_back(steps.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&seq_output));
    seq_task_data->outputs_count.emplace_back(1);

    auto seq_task = SequentialTask(seq_task_data, func);
    ASSERT_TRUE(seq_task.validation());
    ASSERT_TRUE(seq_task.pre_processing());
    ASSERT_TRUE(seq_task.run());
    ASSERT_TRUE(seq_task.post_processing());
    EXPECT_NEAR(seq_output, par_output, tolerance);
  }
}

void run_invalid_task(func_nd_t func, bounds_t& bounds, steps_t& steps) {
  auto world = boost::mpi::communicator();

  if (world.rank() == 0) {
    auto par_output = 0.0;
    auto par_task_data = std::make_shared<ppc::core::TaskData>();
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds.data()));
    par_task_data->inputs_count.emplace_back(bounds.size());
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
    par_task_data->inputs_count.emplace_back(steps.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&par_output));
    par_task_data->outputs_count.emplace_back(1);

    auto par_task = ParallelTask(par_task_data, func);
    ASSERT_FALSE(par_task.validation());
  }
}

}  // namespace chernykh_a_multidimensional_integral_simpson_mpi

namespace chernykh_a_mis_mpi = chernykh_a_multidimensional_integral_simpson_mpi;

TEST(chernykh_a_multidimensional_integral_simpson_mpi, linear_2d_integration) {
  auto func = [](const auto& args) { return (2 * args[0]) + (3 * args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 1.0}, {0.0, 2.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, quadratic_3d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]) + (args[2] * args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2, 2};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, exponential_3d_integration) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1] + args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}};
  auto steps = chernykh_a_mis_mpi::steps_t{4, 4, 4};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, trigonometric_2d_integration) {
  auto func = [](const auto& args) { return std::sin(args[0]) * std::cos(args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}};
  auto steps = chernykh_a_mis_mpi::steps_t{20, 20};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, polynomial_3d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[1]) + (args[1] * args[2]) + (args[0] * args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2, 2};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, linear_3d_integration) {
  auto func = [](const auto& args) { return args[0] + (2 * args[1]) + (3 * args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 2.0}, {0.0, 1.0}, {0.0, 3.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2, 2};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, quadratic_2d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 2.0}, {0.0, 3.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, exponential_2d_integration) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{10, 10};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, trigonometric_3d_integration) {
  auto func = [](const auto& args) { return std::sin(args[0]) * std::cos(args[1]) * std::tan(args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{
      {0.0, std::numbers::pi},
      {0.0, std::numbers::pi / 2},
      {0.0, std::numbers::pi / 4},
  };
  auto steps = chernykh_a_mis_mpi::steps_t{16, 16, 16};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, cubic_2d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0] * args[0]) + (args[1] * args[1] * args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 1.0}, {0.0, 2.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  auto tolerance = 1e-5;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, small_range_linear_2d_integration) {
  auto func = [](const auto& args) { return (2 * args[0]) + (3 * args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.00000001, 0.00000003}, {0.00000002, 0.00000004}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  auto tolerance = 1e-23;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, small_range_quadratic_2d_integration) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.00000005, 0.00000006}, {0.00000001, 0.00000002}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  auto tolerance = 1e-32;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, small_range_exponential_2d_integration) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.00000003, 0.00000005}, {0.00000002, 0.00000004}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  auto tolerance = 1e-16;
  chernykh_a_mis_mpi::run_valid_task(func, bounds, steps, tolerance);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, empty_bounds_fails_validation) {
  auto func = [](const auto& args) { return (2 * args[0]) + (3 * args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  chernykh_a_mis_mpi::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, incorrect_bounds_fails_validation) {
  auto func = [](const auto& args) { return std::exp(args[0] + args[1] + args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 0.5}, {1.0, 0.5}, {0.0, 0.5}};
  auto steps = chernykh_a_mis_mpi::steps_t{4, 4, 4};
  chernykh_a_mis_mpi::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, empty_steps_fails_validation) {
  auto func = [](const auto& args) { return (args[0] * args[0]) + (args[1] * args[1]) + (args[2] * args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{};
  chernykh_a_mis_mpi::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, incorrect_steps_fails_validation) {
  auto func = [](const auto& args) { return std::sin(args[0]) * std::cos(args[1]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}};
  auto steps = chernykh_a_mis_mpi::steps_t{20, 21};
  chernykh_a_mis_mpi::run_invalid_task(func, bounds, steps);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, bounds_steps_size_mismatch_fails_validation) {
  auto func = [](const auto& args) { return (args[0] * args[1]) + (args[1] * args[2]) + (args[0] * args[2]); };
  auto bounds = chernykh_a_mis_mpi::bounds_t{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps = chernykh_a_mis_mpi::steps_t{2, 2};
  chernykh_a_mis_mpi::run_invalid_task(func, bounds, steps);
}
