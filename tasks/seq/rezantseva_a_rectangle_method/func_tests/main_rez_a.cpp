// FUNC_TEST_SEQ_RECTANGLE
#include <gtest/gtest.h>

#include "seq/rezantseva_a_rectangle_method/include/ops_seq_rez_a.hpp"

TEST(rezantseva_a_rectangle_method_seq, check_1_dimension_integral) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0];  // x^2
  };

  int n = 1;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 10};
  distrib[0] = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_rectangle_method_seq::RectangleMethodSequential RectangleMethodSequential(taskDataSeq, function);
  ASSERT_EQ(RectangleMethodSequential.validation(), true);
  RectangleMethodSequential.pre_processing();
  RectangleMethodSequential.run();
  RectangleMethodSequential.post_processing();
  double error = 0.0001;
  ASSERT_NEAR(312, out[0], error);
}

TEST(rezantseva_a_rectangle_method_seq, check_linear_func) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0] - 2 * x[1];  // x^2-2y
  };

  int n = 2;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 10};
  bounds[1] = {1, 56};
  distrib[0] = 1000;
  distrib[1] = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_rectangle_method_seq::RectangleMethodSequential RectangleMethodSequential(taskDataSeq, function);
  ASSERT_EQ(RectangleMethodSequential.validation(), true);
  RectangleMethodSequential.pre_processing();
  RectangleMethodSequential.run();
  RectangleMethodSequential.post_processing();
  double error = 0.001;
  ASSERT_NEAR(-1650, out[0], error);
}

TEST(rezantseva_a_rectangle_method_seq, check_1_dimension_integral_sin) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return std::sin(x[0]) + x[0] * x[0] * x[0];  // sinx+x^3
  };

  int n = 1;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 10};
  distrib[0] = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_rectangle_method_seq::RectangleMethodSequential RectangleMethodSequential(taskDataSeq, function);
  ASSERT_EQ(RectangleMethodSequential.validation(), true);
  RectangleMethodSequential.pre_processing();
  RectangleMethodSequential.run();
  RectangleMethodSequential.post_processing();
  double error = 0.001;
  ASSERT_NEAR(2436.18542, out[0], error);
}
TEST(rezantseva_a_rectangle_method_seq, check_2_dimension_integral_sin) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return std::sin(x[0]) + x[0] * x[0] * x[1];  // sinx + y*x^2
  };

  int n = 2;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {-3, 10};
  bounds[1] = {-7, 25};
  distrib[0] = 10000;
  distrib[1] = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_rectangle_method_seq::RectangleMethodSequential RectangleMethodSequential(taskDataSeq, function);
  ASSERT_EQ(RectangleMethodSequential.validation(), true);
  RectangleMethodSequential.pre_processing();
  RectangleMethodSequential.run();
  RectangleMethodSequential.post_processing();
  double error = 0.001;
  ASSERT_NEAR(98587.1705290392, out[0], error);
}

TEST(rezantseva_a_rectangle_method_seq, check_3_dimension_integral) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] + x[1] + x[2];  // xyz
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 100};
  bounds[1] = {1, 156};
  bounds[2] = {6, 249};
  distrib[0] = 100;
  distrib[1] = 100;
  distrib[2] = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_rectangle_method_seq::RectangleMethodSequential RectangleMethodSequential(taskDataSeq, function);
  ASSERT_EQ(RectangleMethodSequential.validation(), true);
  RectangleMethodSequential.pre_processing();
  RectangleMethodSequential.run();
  RectangleMethodSequential.post_processing();
  double error = 0.001;
  ASSERT_NEAR(932886720.0, out[0], error);
}

TEST(rezantseva_a_rectangle_method_seq, check_3_dimension_integral_with_exp) {
  std::vector<double> out(1, 0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return exp(x[0] + 2 * x[1]) - 2 * cos(x[2]);  // exp(x+2y) - 2cosz + sqrt(4d)
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 10};
  bounds[1] = {-7, 3};
  bounds[2] = {3, 8};

  distrib[0] = 500;
  distrib[1] = 250;
  distrib[2] = 150;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
  taskDataSeq->inputs_count.emplace_back(bounds.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
  taskDataSeq->inputs_count.emplace_back(distrib.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_rectangle_method_seq::RectangleMethodSequential RectangleMethodSequential(taskDataSeq, function);
  ASSERT_EQ(RectangleMethodSequential.validation(), true);
  RectangleMethodSequential.pre_processing();
  RectangleMethodSequential.run();
  RectangleMethodSequential.post_processing();
  double error = 0.1;
  ASSERT_NEAR(22154067.07, out[0], error);
}
