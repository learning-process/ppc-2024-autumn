// Copyright 2023 Nesterov Alexander
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <random>
#include <vector>

#include "seq/ermolaev_v_multidimensional_integral_rectangle/include/ops_seq.hpp"

namespace ermolaev_v_multidimensional_integral_rectangle_seq {

void testBody(std::vector<std::pair<double, double>> limits, double ref,
              ermolaev_v_multidimensional_integral_rectangle_seq::function func, double eps = 1e-4) {
  double out = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  ermolaev_v_multidimensional_integral_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq, func);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(ref, out, eps);
}

double simpleOneVar(std::vector<double>& args) { return args.at(0); }
double simpleTwoVar(std::vector<double>& args) { return args.at(0) + args.at(1); }
double simpleThreeVar(std::vector<double>& args) { return args.at(0) + args.at(1) + args.at(2); }
double advancedOneVar(std::vector<double>& args) {
  return std::sin(args.at(0)) / (std::pow(std::cos(args.at(0)), 2) + 1);
}
double advancedTwoVar(std::vector<double>& args) { return std::tan(args.at(0)) - std::cos(args.at(1)); }
double advancedThreeVar(std::vector<double>& args) {
  return std::sin(args.at(1)) + std::cos(args.at(0)) - exp(args.at(2));
}

}  // namespace ermolaev_v_multidimensional_integral_rectangle_seq
namespace erm_integral_seq = ermolaev_v_multidimensional_integral_rectangle_seq;

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_double_integral_one_variable) {
  erm_integral_seq::testBody({{0, 2}, {0, 2}}, 4, erm_integral_seq::simpleOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_triple_integral_one_variable) {
  erm_integral_seq::testBody({{0, 2}, {0, 2}, {0, 2}}, 8, erm_integral_seq::simpleOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_quad_integral_one_variable) {
  erm_integral_seq::testBody({{0, 2}, {0, 2}, {0, 2}, {0, 2}}, 16, erm_integral_seq::simpleOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_8th_integral_one_variable) {
  int32_t dimension = 8;
  std::vector<std::pair<double, double>> limits(dimension, {0, 2});
  erm_integral_seq::testBody(limits, std::pow(2, dimension), erm_integral_seq::simpleOneVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_double_integral_two_variables) {
  erm_integral_seq::testBody({{0, 2}, {0, 2}}, 8, erm_integral_seq::simpleTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_triple_integral_two_variables) {
  erm_integral_seq::testBody({{0, 2}, {0, 2}, {0, 2}}, 16, erm_integral_seq::simpleTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_quad_integral_two_variables) {
  erm_integral_seq::testBody({{0, 2}, {0, 2}, {0, 2}, {0, 2}}, 32, erm_integral_seq::simpleTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_8th_integral_two_variables) {
  int32_t dimension = 8;
  std::vector<std::pair<double, double>> limits(dimension, {0, 2});
  erm_integral_seq::testBody(limits, std::pow(2, dimension + 1), erm_integral_seq::simpleTwoVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_triple_integral_three_variables) {
  erm_integral_seq::testBody({{1, 3}, {1, 3}, {1, 3}}, 48, erm_integral_seq::simpleThreeVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_quad_integral_three_variables) {
  erm_integral_seq::testBody({{1, 3}, {1, 3}, {1, 3}, {1, 3}}, 96, erm_integral_seq::simpleThreeVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, simple_8th_integral_three_variables) {
  int32_t dimension = 8;
  std::vector<std::pair<double, double>> limits(dimension, {1, 3});
  erm_integral_seq::testBody(limits, std::pow(2, dimension + 2), erm_integral_seq::simpleTwoVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, advanced_double_integral_one_variable) {
  erm_integral_seq::testBody({{-5, 2}, {3, 6}}, 2.01228, erm_integral_seq::advancedOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, advanced_triple_integral_one_variable) {
  erm_integral_seq::testBody({{-5, 2}, {3, 6}, {21, 22.5}}, 3.01842, erm_integral_seq::advancedOneVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, advanced_double_integral_two_variables) {
  erm_integral_seq::testBody({{-0.5, 0.8}, {-2, 2}}, -1.44106, erm_integral_seq::advancedTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_seq, advanced_triple_integral_two_variables) {
  erm_integral_seq::testBody({{-0.5, 0.8}, {-2, 2}, {2.5, 2.6}}, -0.14421, erm_integral_seq::advancedTwoVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, advanced_triple_integral_three_variables) {
  erm_integral_seq::testBody({{-0.5, 0.8}, {-2, 2}, {2.5, 2.6}}, -6.18355, erm_integral_seq::advancedThreeVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_seq, validation) {
  const auto validate = [](std::shared_ptr<ppc::core::TaskData>& taskData) {
    ermolaev_v_multidimensional_integral_rectangle_seq::function func = erm_integral_seq::simpleOneVar;
    ermolaev_v_multidimensional_integral_rectangle_seq::TestTaskSequential task(taskData, func);
    return task.validation();
  };

  std::vector<std::pair<double, double>> limits{{0, 2}, {0, 2}};
  double eps = 1e-4;
  double out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(0);
  ASSERT_FALSE(validate(taskData));
  taskData->inputs_count.front() = limits.size();
  ASSERT_FALSE(validate(taskData));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
  ASSERT_FALSE(validate(taskData));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskData->outputs_count.emplace_back(0);
  ASSERT_FALSE(validate(taskData));
  taskData->outputs_count.front() = 1;
  ASSERT_FALSE(validate(taskData));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
  ASSERT_TRUE(validate(taskData));
}
