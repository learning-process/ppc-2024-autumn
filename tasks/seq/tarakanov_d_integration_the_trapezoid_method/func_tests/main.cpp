// Copyright 2024 Tarakanov Denis
#include <gtest/gtest.h>

#include <vector>

#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, Test_Integration_the_trapezoid_method_1) {
  double a = 0;
  double b = 1;
  double h = 0.1;

  // Create data
  std::vector<double> in(3, 0);
  in[0] = a;
  in[1] = b;
  in[2] = h;

  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 3);
}

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, Test_Integration_the_trapezoid_method_2) {
  double a = 0;
  double b = 2;
  double h = 0.1;

  // Create data
  std::vector<double> in(3, 0);
  in[0] = a;
  in[1] = b;
  in[2] = h;

  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 2);
}