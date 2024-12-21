// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "seq/fyodorov_m_trapezoidal_method_seq/include/ops_seq.hpp"

using namespace fyodorov_m_trapezoidal_method_seq;

namespace {

bool almost_equal(double a, double b, double epsilon = 1e-6) { return std::abs(a - b) < epsilon; }

double test_func_2(const std::vector<double> &x) { return x[0]; }

double test_func_3(const std::vector<double> &x) { return x[0] + x[1] + x[2]; }

// double test_func_4(const std::vector<double> &x) { return std::sin(x[0]); }

// double test_func_5(const std::vector<double> &x) { return x[0] * x[0] * x[0]; }

double test_func_6(const std::vector<double> &x) { return x[0] * x[1]; }

// double test_func_7(const std::vector<double> &x) {
//   (void)x;
//   return 1.0;
// }

// double test_func_10(const std::vector<double> &x) { return x[0] / 2.0; }

double test_func_12(const std::vector<double> &x) { return x[0] * x[1] * x[2]; }
}  // namespace

TEST(Sequential, Test_1D_Linear) {
  // Create data
  std::function<double(const std::vector<double> &)> func = test_func_2;
  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {1.0};
  std::vector<int> intervals = {100};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs_count.emplace_back(intervals.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(almost_equal(out[0], 0.5));
}

TEST(Sequential, Test_2D_Linear) {
  // Create data
  std::function<double(const std::vector<double> &)> func = test_func_6;
  std::vector<double> lower_bounds = {0.0, 0.0};
  std::vector<double> upper_bounds = {1.0, 1.0};
  std::vector<int> intervals = {100, 100};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs_count.emplace_back(intervals.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(almost_equal(out[0], 0.25));
}

TEST(Sequential, Test_3D_Linear) {
  // Create data
  std::function<double(const std::vector<double> &)> func = test_func_3;
  std::vector<double> lower_bounds = {0.0, 0.0, 0.0};
  std::vector<double> upper_bounds = {1.0, 1.0, 1.0};
  std::vector<int> intervals = {50, 50, 50};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs_count.emplace_back(intervals.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(almost_equal(out[0], 1.5));
}

TEST(Sequential, Test_2D_Product) {
  // Create data
  std::function<double(const std::vector<double> &)> func = test_func_6;
  std::vector<double> lower_bounds = {0.0, 0.0};
  std::vector<double> upper_bounds = {1.0, 1.0};
  std::vector<int> intervals = {100, 100};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs_count.emplace_back(intervals.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(almost_equal(out[0], 0.25, 1e-5));
}

TEST(Sequential, Test_3D_Product) {
  // Create data
  std::function<double(const std::vector<double> &)> func = test_func_12;
  std::vector<double> lower_bounds = {0.0, 0.0, 0.0};
  std::vector<double> upper_bounds = {1.0, 1.0, 1.0};
  std::vector<int> intervals = {10, 10, 10};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lower_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(upper_bounds.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs_count.emplace_back(intervals.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(almost_equal(out[0], 1.0 / 8.0, 1e-5));
}
/*
TEST(Sequential, Test_Sum_10) {
  // Create data
  std::vector<int> in(1, 2, 3, 4);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  fyodorov_m_trapezoidal_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(Sequential, Test_Sum_20) {
  const int count = 20;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  fyodorov_m_trapezoidal_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(Sequential, Test_Sum_50) {
  const int count = 50;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  fyodorov_m_trapezoidal_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(Sequential, Test_Sum_70) {
  const int count = 70;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  fyodorov_m_trapezoidal_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(Sequential, Test_Sum_100) {
  const int count = 100;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  fyodorov_m_trapezoidal_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
*/
