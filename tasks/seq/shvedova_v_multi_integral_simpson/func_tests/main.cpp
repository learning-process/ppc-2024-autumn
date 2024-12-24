#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

#include "seq/shvedova_v_multi_integral_simpson/include/ops_seq.hpp"

namespace shvedova_v_multidimensional_integral_simpson_seq {

double linearFunction(std::vector<double>& args) { return args[0]; }
double sumTwoVariables(std::vector<double>& args) { return args[0] + args[1]; }
double quadraticFunction(std::vector<double>& args) { return args[0] * args[0]; }
double productTwoVariables(std::vector<double>& args) { return args[0] * args[1]; }
double trigonometricFunction(std::vector<double>& args) { return std::sin(args[0]) + std::cos(args[1]); }
double exponentialFunction(std::vector<double>& args) { return exp(args[0]) - args[1]; }
double tripleLinearFunction(std::vector<double>& args) { return args[0] + args[1] + args[2]; }
double tripleProductFunction(std::vector<double>& args) { return args[0] * args[1] * args[2]; }

void runTest(std::vector<std::pair<double, double>> limits, double expectedValue,
             shvedova_v_multidimensional_integral_simpson_seq::FunctionType testFunction, double precision = 1e-3) {
  double result = 0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&testFunction));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
  taskData->inputs_count.emplace_back(limits.size());

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.emplace_back(1);

  shvedova_v_multidimensional_integral_simpson_seq::MultidimensionalIntegralSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  EXPECT_NEAR(expectedValue, result, precision);
}

}  // namespace shvedova_v_multidimensional_integral_simpson_seq

TEST(shvedova_v_multidimensional_integral_simpson_seq, LinearFunction_SingleVariable) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{0, 2}}, 2.0, shvedova_v_multidimensional_integral_simpson_seq::linearFunction);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, LinearFunction_MultipleVariables) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{0, 2}, {0, 2}}, 4.0, shvedova_v_multidimensional_integral_simpson_seq::linearFunction);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, SumTwoVariables_TwoDimensions) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{0, 2}, {0, 2}}, 8.0, shvedova_v_multidimensional_integral_simpson_seq::sumTwoVariables);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, QuadraticFunction_OneDimension) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{0, 3}}, 9.0, shvedova_v_multidimensional_integral_simpson_seq::quadraticFunction);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, ProductTwoVariables_TwoDimensions) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{1, 3}, {2, 4}}, 24.0, shvedova_v_multidimensional_integral_simpson_seq::productTwoVariables);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, TrigonometricFunction_TwoDimensions) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{0, M_PI}, {0, M_PI / 2}}, 2 * M_PI, shvedova_v_multidimensional_integral_simpson_seq::trigonometricFunction);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, ExponentialFunction_TwoDimensions) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{0, 1}, {0, 1}}, std::numbers::e - 1.5, shvedova_v_multidimensional_integral_simpson_seq::exponentialFunction);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, TripleLinearFunction) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{0, 1}, {0, 1}, {0, 1}}, 1.5, shvedova_v_multidimensional_integral_simpson_seq::tripleLinearFunction);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, TripleProductFunction) {
  shvedova_v_multidimensional_integral_simpson_seq::runTest(
      {{1, 2}, {1, 2}, {1, 2}}, 3.375, shvedova_v_multidimensional_integral_simpson_seq::tripleProductFunction);
}
