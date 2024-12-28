#include <gtest/gtest.h>

#include "seq/korneeva_e_rectangular_integration_method/include/ops_seq.hpp"

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

korneeva_e_rectangular_integration_method_seq::RectangularIntegration createIntegrationTask(
    const std::shared_ptr<ppc::core::TaskData>& taskData,
    const std::function<double(const std::vector<double>&)>& integrand) {
  return korneeva_e_rectangular_integration_method_seq::RectangularIntegration(taskData, integrand);
}

TEST(korneeva_e_rectangular_integration_method_seq, invalid_limits) {
  std::vector<std::pair<double, double>> lims = {{1.0, 0.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_FALSE(task.validation());
}

TEST(korneeva_e_rectangular_integration_method_seq, InvalidEpsilonUsedDefault) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}};
  double epsilon = -1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.84147, out[0], 1e-4);
}

TEST(korneeva_e_rectangular_integration_method_seq, InvalidNumDimensions) {
  const size_t dim = 0;
  std::vector<std::pair<double, double>> lims(dim);
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_FALSE(task.validation());
}

TEST(korneeva_e_rectangular_integration_method_seq, InvalidNumOutputs) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(2);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_FALSE(task.validation());
}

// Test for integrating cos(x) over the range [-0.5*pi, 0.5*pi]
TEST(korneeva_e_rectangular_integration_method_seq, CosFunctionOverSymmetricInterval) {
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

  ASSERT_NEAR(2.0, out[0], epsilon);
}

// Test for integrating x^2 over [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, PolynomialFunctionOverUnitInterval) {
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

  ASSERT_NEAR(1.0 / 3.0, out[0], epsilon);
}

// Test for integrating x + y over [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, LinearFunctionOverUnitSquare) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f2x_squared_plus_y_squared = [](const std::vector<double>& args) { return args[0] + args[1]; };
  auto task = createIntegrationTask(taskData, f2x_squared_plus_y_squared);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(1.0, out[0], epsilon);
}

// Test for very small epsilon
TEST(korneeva_e_rectangular_integration_method_seq, VerySmallEpsilon) {
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

  ASSERT_NEAR(0.5, out[0], 1e-6);
}

// Test for very large epsilon (precision = 1)
TEST(korneeva_e_rectangular_integration_method_seq, VeryLargeEpsilon) {
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

  ASSERT_NEAR(0.84147, out[0], 0.1);
}

// Test for 3-dimensional integral of x + y + z over [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, LinearFunctionOverUnitCube) {
  std::vector<std::pair<double, double>> lims = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f3d = [](const std::vector<double>& args) { return args[0] + args[1] + args[2]; };
  auto task = createIntegrationTask(taskData, f3d);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(1.5, out[0], epsilon);  // Verify result with epsilon precision
}

// Test for 3-dimensional integral of sin(x) * cos(y) * exp(z) over [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, TrigonometricExponentialFunctionOverUnitCube) {
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

  ASSERT_NEAR(0.664697, out[0], epsilon);
}

// Test for 4-dimensional integral of x + y + z + w over [0, 1] x [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, LinearFunctionOverUnitHypercube) {
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

  ASSERT_NEAR(2.0, out[0], epsilon);
}

// Test for 5-dimensional integral of x + y + z + w + v over [0, 1] x [0, 1] x [0, 1] x [0, 1] x [0, 1]
TEST(korneeva_e_rectangular_integration_method_seq, LinearFunctionOverUnit5DHypercube) {
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

  ASSERT_NEAR(2.5, out[0], epsilon);
}

TEST(korneeva_e_rectangular_integration_method_seq, ZeroFunction) {
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

  ASSERT_NEAR(0.0, out[0], epsilon);
}

TEST(korneeva_e_rectangular_integration_method_seq, ZeroInterval) {
  std::vector<std::pair<double, double>> lims = {{1.0, 1.0}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1cos = [](const std::vector<double>& args) { return std::cos(args[0]); };
  auto task = createIntegrationTask(taskData, f1cos);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(0.0, out[0], epsilon);
}

TEST(korneeva_e_rectangular_integration_method_seq, ConstantFunction) {
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

  ASSERT_NEAR(1.0, out[0], epsilon);
}

TEST(korneeva_e_rectangular_integration_method_seq, LargeLimitsIntegration) {
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
  ASSERT_NEAR(expectedValue, out[0], epsilon);
}

TEST(korneeva_e_rectangular_integration_method_seq, SmallRange) {
  std::vector<std::pair<double, double>> lims = {{0.0000001, 0.0000002}};
  double epsilon = 1e-4;
  std::vector<double> out(1);

  auto taskData = prepareTaskData(lims, &epsilon, out);
  auto f1x = [](const std::vector<double>& args) { return args[0]; };
  auto task = createIntegrationTask(taskData, f1x);

  ASSERT_TRUE(task.validation());

  task.pre_processing();
  task.run();
  task.post_processing();

  double expected_value = 0.00000015;
  ASSERT_NEAR(expected_value, out[0], epsilon);
}
