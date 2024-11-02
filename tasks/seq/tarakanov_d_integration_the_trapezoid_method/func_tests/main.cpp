#include <gtest/gtest.h>

#include <memory>

#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

using namespace tarakanov_d_integration_the_trapezoid_method_seq;

auto createTaskData(double* a, double* b, double* h, double* res) {
  auto data = std::make_shared<ppc::core::TaskData>();

  data->inputs.push_back(reinterpret_cast<uint8_t*>(a));
  data->inputs.push_back(reinterpret_cast<uint8_t*>(b));
  data->inputs.push_back(reinterpret_cast<uint8_t*>(h));

  data->inputs_count.push_back(3);

  data->outputs.push_back(reinterpret_cast<uint8_t*>(res));
  data->outputs_count.push_back(1);

  return data;
}

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, ValidationWorks) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = createTaskData(&a, &b, &h, &res);

  integration_the_trapezoid_method task(data);

  EXPECT_TRUE(task.validation());
}

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, PreProcessingWorks) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = createTaskData(&a, &b, &h, &res);
  integration_the_trapezoid_method task(data);

  EXPECT_TRUE(task.validation());
  EXPECT_TRUE(task.pre_processing());
  EXPECT_EQ(task.get_data()->inputs_count[0], 3.0);
  EXPECT_EQ(task.get_data()->outputs_count[0], 1.0);
}

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, PostProcessingWorks) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = createTaskData(&a, &b, &h, &res);

  integration_the_trapezoid_method task(data);
  EXPECT_TRUE(task.validation());
  EXPECT_TRUE(task.pre_processing());
  EXPECT_TRUE(task.run());
  EXPECT_TRUE(task.post_processing());

  double output = *reinterpret_cast<double*>(data->outputs[0]);
  bool flag = output == 0.0;
  EXPECT_FALSE(flag);
}
