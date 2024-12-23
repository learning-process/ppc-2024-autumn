// Copyright 2023 Nesterov Alexander
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/ermolaev_v_multidimensional_integral_rectangle/include/ops_mpi.hpp"

namespace ermolaev_v_multidimensional_integral_rectangle_mpi {

void testBody(std::vector<std::pair<double, double>> limits,
              ermolaev_v_multidimensional_integral_rectangle_mpi::function func, double eps = 1e-4) {
  boost::mpi::communicator world;
  double out = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(limits.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Create Task
  ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskParallel testTaskParallel(taskDataPar, func);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    double seq_out = 0;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(limits.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&seq_out));
    taskDataSeq->outputs_count.emplace_back(1);

    ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq, func);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(seq_out, out, eps);
  }
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

}  // namespace ermolaev_v_multidimensional_integral_rectangle_mpi
namespace erm_integral_mpi = ermolaev_v_multidimensional_integral_rectangle_mpi;

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_double_integral_one_variable) {
  erm_integral_mpi::testBody({{0, 2}, {0, 2}}, erm_integral_mpi::simpleOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_triple_integral_one_variable) {
  erm_integral_mpi::testBody({{0, 2}, {0, 2}, {0, 2}}, erm_integral_mpi::simpleOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_quad_integral_one_variable) {
  erm_integral_mpi::testBody({{0, 2}, {0, 2}, {0, 2}, {0, 2}}, erm_integral_mpi::simpleOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_8th_integral_one_variable) {
  int32_t dimension = 8;
  std::vector<std::pair<double, double>> limits(dimension, {0, 2});
  erm_integral_mpi::testBody(limits, erm_integral_mpi::simpleOneVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_double_integral_two_variables) {
  erm_integral_mpi::testBody({{0, 2}, {0, 2}}, erm_integral_mpi::simpleTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_triple_integral_two_variables) {
  erm_integral_mpi::testBody({{0, 2}, {0, 2}, {0, 2}}, erm_integral_mpi::simpleTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_quad_integral_two_variables) {
  erm_integral_mpi::testBody({{0, 2}, {0, 2}, {0, 2}, {0, 2}}, erm_integral_mpi::simpleTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_8th_integral_two_variables) {
  int32_t dimension = 8;
  std::vector<std::pair<double, double>> limits(dimension, {0, 2});
  erm_integral_mpi::testBody(limits, erm_integral_mpi::simpleTwoVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_triple_integral_three_variables) {
  erm_integral_mpi::testBody({{1, 3}, {1, 3}, {1, 3}}, erm_integral_mpi::simpleThreeVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_quad_integral_three_variables) {
  erm_integral_mpi::testBody({{1, 3}, {1, 3}, {1, 3}, {1, 3}}, erm_integral_mpi::simpleThreeVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, simple_8th_integral_three_variables) {
  int32_t dimension = 8;
  std::vector<std::pair<double, double>> limits(dimension, {1, 3});
  erm_integral_mpi::testBody(limits, erm_integral_mpi::simpleTwoVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, advanced_double_integral_one_variable) {
  erm_integral_mpi::testBody({{-5, 2}, {3, 6}}, erm_integral_mpi::advancedOneVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, advanced_triple_integral_one_variable) {
  erm_integral_mpi::testBody({{-5, 2}, {3, 6}, {21, 22.5}}, erm_integral_mpi::advancedOneVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, advanced_double_integral_two_variables) {
  erm_integral_mpi::testBody({{-0.5, 0.8}, {-2, 2}}, erm_integral_mpi::advancedTwoVar);
}
TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, advanced_triple_integral_two_variables) {
  erm_integral_mpi::testBody({{-0.5, 0.8}, {-2, 2}, {2.5, 2.6}}, erm_integral_mpi::advancedTwoVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, advanced_triple_integral_three_variables) {
  erm_integral_mpi::testBody({{-0.5, 0.8}, {-2, 2}, {2.5, 2.6}}, erm_integral_mpi::advancedThreeVar);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, validation) {
  const auto validate = [](std::shared_ptr<ppc::core::TaskData>& taskData) {
    ermolaev_v_multidimensional_integral_rectangle_mpi::function func = erm_integral_mpi::simpleOneVar;
    ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskParallel task(taskData, func);
    return task.validation();
  };

  boost::mpi::communicator world;
  std::vector<std::pair<double, double>> limits{{0, 2}, {0, 2}};
  double eps = 1e-4;
  double out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
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
}
