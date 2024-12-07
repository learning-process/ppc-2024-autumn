#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/smirnov_i_integration_by_rectangles/include/ops_seq.hpp"
double f1(double x) { return x * x; }
double f2(double x) { return std::exp(x); }
double f3(double x) { return std::sin(x); }
double f_const(double x) { return 5 + 0 * x; }
double f_lin(double x) { return x; }
TEST(smirnov_i_integration_by_rectangles_seq, Test_invalid_func) {
  double left = 0;
  double right = 1;
  int n = 1000;
  std::vector<double> res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

  testMpiTaskSequential.set_function(nullptr);

  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  ASSERT_ANY_THROW(testMpiTaskSequential.run());
}
TEST(smirnov_i_integration_by_rectangles_seq, Test_const) {
  double left = 0;
  double right = 1;
  int n = 1000;
  double expected_result = 5;
  std::vector<double> res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

  testMpiTaskSequential.set_function(f_const);

  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_NEAR(res[0], expected_result, 1e-6);
}
TEST(smirnov_i_integration_by_rectangles_seq, Test_linear) {
  double left = 0;
  double right = 1;
  int n = 1000;
  double expected_result = 1. / 2;
  std::vector<double> res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

  testMpiTaskSequential.set_function(f_lin);

  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_NEAR(expected_result, res[0], 1e-6);
}
TEST(smirnov_i_integration_by_rectangles_seq, Test_x_times_x) {
  double left = 0;
  double right = 1;
  int n = 1000;
  double expected_result = 1. / 3;
  std::vector<double> res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

  testMpiTaskSequential.set_function(f1);

  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_NEAR(expected_result, res[0], 1e-6);
}
TEST(smirnov_i_integration_by_rectangles_seq, Test_e_x) {
  double left = 0;
  double right = 1;
  int n = 1000;
  double expected_result = std::exp(1.0) - 1;
  std::vector<double> res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

  testMpiTaskSequential.set_function(f2);

  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_NEAR(expected_result, res[0], 1e-6);
}

TEST(smirnov_i_integration_by_rectangles_seq, Test_sin_x) {
  double left = 0;
  double right = 1;
  int n = 1000;
  double expected_result = 1 - std::cos(1);
  std::vector<double> res(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

  testMpiTaskSequential.set_function(f3);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_NEAR(expected_result, res[0], 1e-6);
}
