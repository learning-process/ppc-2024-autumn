#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "seq/shulpin_i_simpson_method/include/simpson_method.hpp"

constexpr double ESTIMATE = 1e-5;

TEST(shulpin_simpson_method, x_plus_y_2n) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 100;

  double global_integral = 0.0;
  double ref_integral = 45.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_x_plus_y);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, x_mul_y_2n) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 100;

  double global_integral = 0.0;
  double ref_integral = 33.75;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_x_mul_y);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, sin_plus_cos_2n) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 100;

  double global_integral = 0.0;
  double ref_integral = 10.20396;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_sin_plus_cos);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, sin_mul_cos_2n) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 100;

  double global_integral = 0.0;
  double ref_integral = 0.16848;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_sin_mul_cos);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, x_plus_y_2n_plus_1) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 101;

  double global_integral = 0.0;
  double ref_integral = 45.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_x_plus_y);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, x_mul_y_2n_plus_1) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 101;

  double global_integral = 0.0;
  double ref_integral = 33.75;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_x_mul_y);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, sin_plus_cos_2n_plus_1) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 101;

  double global_integral = 0.0;
  double ref_integral = 10.20396;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_sin_plus_cos);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, sin_mul_cos_2n_plus_1) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = 4.0;

  int N = 101;

  double global_integral = 0.0;
  double ref_integral = 0.16848;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq seqIntegral(taskDataSeq);
  seqIntegral.set_seq(shulpin_simpson_method::f_sin_mul_cos);
  ASSERT_EQ(seqIntegral.validation(), true);
  seqIntegral.pre_processing();
  seqIntegral.run();
  seqIntegral.post_processing();

  ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
}

TEST(shulpin_simpson_method, miss_of_data) {
  double a = 0.0;
  double b = 3.0;
  double c = -1.0;
  double d = c + 2.0;

  int N = 101;

  double global_integral = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq IntegralSeq(taskDataSeq);
  ASSERT_FALSE(IntegralSeq.validation());
}

TEST(shulpin_simpson_method, invalid_data_1) {
  double a = 3.0;
  double b = 0.0;
  double c = 1.0;
  double d = 2.0;

  int N = 101;

  double global_integral = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq IntegralSeq(taskDataSeq);
  ASSERT_FALSE(IntegralSeq.validation());
}

TEST(shulpin_simpson_method, invalid_data_2) {
  double a = 0.0;
  double b = 3.0;
  double c = 2.0;
  double d = 1.0;

  int N = 101;

  double global_integral = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq IntegralSeq(taskDataSeq);

  ASSERT_FALSE(IntegralSeq.validation());
}

TEST(shulpin_simpson_method, invalid_data_3) {
  double a = 0.0;
  double b = 3.0;
  double c = 0.0;
  double d = 4.0;

  int N = -101;

  double global_integral = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&c));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&d));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
  taskDataSeq->outputs_count.emplace_back(1);

  shulpin_simpson_method::SimpsonMethodSeq IntegralSeq(taskDataSeq);

  ASSERT_FALSE(IntegralSeq.validation());
}