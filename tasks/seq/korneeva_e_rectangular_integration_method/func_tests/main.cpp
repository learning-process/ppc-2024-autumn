#include <gtest/gtest.h>

#include "seq/korneeva_e_rectangular_integration_method/include/ops_seq.hpp"

// Helper function to prepare TaskData
std::shared_ptr<ppc::core::TaskData> prepareTaskData(const std::vector<std::pair<double, double>>& limits,
                                                     double* epsilon, std::vector<double>& outputs) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<std::pair<double, double>*>(limits.data())));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(epsilon));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputs.data()));
  taskData->outputs_count.emplace_back(outputs.size());
  return taskData;
}

// Helper function to create and validate RectangularIntegration object
korneeva_e_rectangular_integration_method_seq::RectangularIntegration createIntegrationTask(
    const std::shared_ptr<ppc::core::TaskData>& taskData,
    const std::function<double(const std::vector<double>&)>& integrand) {
  return korneeva_e_rectangular_integration_method_seq::RectangularIntegration(taskData, integrand);
}

// Test for invalid integration bounds: minimum value is greater than maximum value
TEST(korneeva_e_rectangular_integration_method_seq, invalid_limits) {
  std::vector<std::pair<double, double>> lims = {{1.0, 0.0}};  // Invalid bounds
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_FALSE(task.validation());  // Validation should fail due to invalid bounds
}

// Test for using default precision when invalid precision is provided
TEST(korneeva_e_rectangular_integration_method_seq, invalid_epsilon_used_default) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}};
  double epsilon = -1e-4;  // Invalid precision (negative)
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_TRUE(task.validation());  // Validation should pass as default precision will be used

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.84147, out[0], 1e-4);  // Verify result with 1e-4 precision
}

// Test for invalid number of dimensions (e.g., zero dimensions)
TEST(korneeva_e_rectangular_integration_method_seq, invalid_num_dimensions) {
  const size_t dim = 0;  // Invalid number of dimensions (zero)
  std::vector<std::pair<double, double>> lims(dim);
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_FALSE(task.validation());  // Validation should fail due to invalid number of dimensions
}

// Test for invalid number of outputs (e.g., two outputs)
TEST(korneeva_e_rectangular_integration_method_seq, invalid_num_outputs) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(2);  // Invalid number of outputs (two)

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_FALSE(task.validation());  // Validation should fail due to invalid number of outputs
}

// Test for integrating cos(x) over the range [-0.5*pi, 0.5*pi]
TEST(korneeva_e_rectangular_integration_method_seq, integrate_cos_over_minus_half_pi_to_half_pi) {
  std::vector<std::pair<double, double>> lims = {{-0.5 * std::numbers::pi, 0.5 * std::numbers::pi}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(2.0, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for integrating x^2 over [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, zero_1_x_squared) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1x_squared = [](const std::vector<double>& args) { return args[0] * args[0]; };
  auto task = createIntegrationTask(taskData, f1x_squared);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(1.0 / 3.0, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for integrating x^2 + y^2 over [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, zero_1_x_1_x_squared_plus_y_squared) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f2x_squared_plus_y_squared = [](const std::vector<double>& args) {
    return args[0] * args[0] + args[1] * args[1];
  };
  auto task = createIntegrationTask(taskData, f2x_squared_plus_y_squared);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(2.0 / 3.0, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for very small epsilon
TEST(korneeva_e_rectangular_integration_method_seq, very_small_epsilon) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}};
  double epsilon = 1e-10;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1x = [](const std::vector<double>& args) { return args[0]; };
  auto task = createIntegrationTask(taskData, f1x);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.5, out[0], 1e-6);  // Verify result with minimal precision
}

// Test for very large epsilon (precision = 1)
TEST(korneeva_e_rectangular_integration_method_seq, very_large_epsilon) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}};
  double epsilon = 1.0;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.84147, out[0], 0.1);  // Verify result with 0.1 precision
}

// Test for 3-dimensional integral of x^2 + y^2 + z^2 over [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq,
     three_dimensional_integral_x_squared_plus_y_squared_plus_z_squared) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f3d = [](const std::vector<double>& args) { return args[0] * args[0] + args[1] * args[1] + args[2] * args[2]; };
  auto task = createIntegrationTask(taskData, f3d);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(1.0, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for 3-dimensional integral of sin(x) * cos(y) * exp(z) over [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, three_dimensional_integral_sin_x_cos_y_exp_z) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f3d = [](const std::vector<double>& args) { return std::sin(args[0]) * std::cos(args[1]) * std::exp(args[2]); };
  auto task = createIntegrationTask(taskData, f3d);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.664697, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for 4-dimensional integral of x + y + z + w over [0, 1] x [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, four_dimensional_integral_x_plus_y_plus_z_plus_w) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f4d = [](const std::vector<double>& args) { return args[0] + args[1] + args[2] + args[3]; };
  auto task = createIntegrationTask(taskData, f4d);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(2.0, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for 5-dimensional integral of x + y + z + w + v over [0, 1] x [0, 1] x [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, five_dimensional_integral_x_plus_y_plus_z_plus_w_plus_v) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f5d = [](const std::vector<double>& args) { return args[0] + args[1] + args[2] + args[3] + args[4]; };
  auto task = createIntegrationTask(taskData, f5d);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(2.5, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for zero function (integration should yield 0)
TEST(korneeva_e_rectangular_integration_method_seq, zero_function) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto fzero = [](const std::vector<double>& args) { return args[0] * 0.0; };
  auto task = createIntegrationTask(taskData, fzero);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.0, out[0], epsilon);  // Verify result is zero for the zero function
}

// Test for zero interval (result should be zero)
TEST(korneeva_e_rectangular_integration_method_seq, zero_interval) {
  std::vector<std::pair<double, double>> lims = {{1.0, 1.0}};  // Degenerate interval
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.0, out[0], epsilon);  // Result should be zero due to degenerate interval
}

// Test for constant function (result should be the area of the integration domain)
TEST(korneeva_e_rectangular_integration_method_seq, constant_function) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto fconstant = [](const std::vector<double>& args) { return std::pow(args[0], 0); };
  auto task = createIntegrationTask(taskData, fconstant);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(1.0, out[0], epsilon);  // Result should be 1 (area of the unit square)
}

// Test for maximum double limits integration (expected result is the range of the integration)
TEST(korneeva_e_rectangular_integration_method_seq, max_double_limits_integration) {
  std::vector<std::pair<double, double>> lims = {
      {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto fconstant = [](const std::vector<double>& args) { return std::pow(args[0], 0); };  // Constant function
  auto task = createIntegrationTask(taskData, fconstant);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  double expectedValue = lims[0].second - lims[0].first;
  ASSERT_EQ(expectedValue, out[0]);  // Verify result matches the expected value (inf)
}

// Test for large but finite limits (integration should yield the range of limits)
TEST(korneeva_e_rectangular_integration_method_seq, large_but_finite_limits_integration) {
  std::vector<std::pair<double, double>> lims = {{-1e6, 1e6}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto fconstant = [](const std::vector<double>& args) { return std::pow(args[0], 0); };
  auto task = createIntegrationTask(taskData, fconstant);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  double expectedValue = lims[0].second - lims[0].first;
  ASSERT_NEAR(expectedValue, out[0], epsilon);  // Verify result matches the expected value (range)
}