#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/prokhorov_n_integral_rectangle_method/include/ops_seq.hpp"

TEST(prokhorov_n_integral_rectangle_method, Test_Integral_x_squared) {
  const double left_ = 0.0;
  const double right_ = 1.0;
  const int n = 1000;
  const double expected_result = 1.0 / 3.0;  

  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return x * x; });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 0.001);
}

TEST(prokhorov_n_integral_rectangle_method, Test_Integral_sin_x) {
  const double left_ = 0.0;
  const double right_ = M_PI;
  const int n = 1000;
  const double expected_result = 2.0;  


  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);


  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());


  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return sin(x); });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();


  ASSERT_NEAR(out[0], expected_result, 0.001);
}


TEST(prokhorov_n_integral_rectangle_method, Test_Integral_exp_x) {
  const double left_ = 0.0;
  const double right_ = 1.0;
  const int n = 1000;
  const double expected_result = exp(1.0) - 1.0; 


  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());


  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return exp(x); });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 0.001);
}


TEST(prokhorov_n_integral_rectangle_method, Test_Integral_arctan_x) {
  const double left_ = 0.0;
  const double right_ = 1.0;
  const int n = 1000;
  const double expected_result = M_PI / 4.0;  


  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);


  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());


  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return 1.0 / (1.0 + x * x); });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 0.001);
}

TEST(prokhorov_n_integral_rectangle_method, Test_Integral_sqrt_x) {
  const double left_ = 0.0;
  const double right_ = 1.0;
  const int n = 1000;
  const double expected_result = 2.0 / 3.0; 

  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return sqrt(x); });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 0.001);
}
TEST(prokhorov_n_integral_rectangle_method, Test_Integral_x_cubed) {
  const double left_ = 0.0;
  const double right_ = 1.0;
  const int n = 1000;
  const double expected_result = 1.0 / 4.0;  

  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return x * x * x; });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 0.001);
}

TEST(prokhorov_n_integral_rectangle_method, Test_Integral_cos_x) {
  const double left_ = 0.0;
  const double right_ = M_PI / 2.0;
  const int n = 1000;
  const double expected_result = 1.0;  

  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return cos(x); });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 0.001);
}

TEST(prokhorov_n_integral_rectangle_method, Test_Integral_log_x) {
  const double left_ = 1.0;
  const double right_ = M_E;
  const int n = 1000;
  const double expected_result = 1.0;  
  std::vector<double> in = {left_, right_, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_integral_rectangle_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_function([](double x) { return log(x); });
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 0.001);
}


