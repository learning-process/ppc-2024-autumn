#include <gtest/gtest.h>

#include <vector>

#include "seq/kholin_k_multidimensional_integrals_rectangle/include/ops_seq.hpp"

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_validation) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_pre_processing) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_run) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_post_processing) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{5.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  ASSERT_EQ(testTaskSequential.post_processing(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, single_integral_one_var) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 1e-2;
  int n = 20;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double ref_I = 0.46;
  ASSERT_NEAR(ref_I, out_I[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, single_integral_two_var) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0, 3.0};
  auto f = [](const std::vector<double> &f_values) { return std::exp(-f_values[0] + f_values[1]); };
  std::vector<double> in_lower_limits{-1};
  std::vector<double> in_upper_limits{5};
  double epsilon = 1e-1;
  int n = 40;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double ref_I = 54.4;
  ASSERT_NEAR(ref_I, out_I[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, double_integral_two_var) {
  // Create data
  size_t dim = 2;
  std::vector<double> values{0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] * f_values[0] + f_values[1] * f_values[1]; };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double epsilon = 1e-3;
  int n = 47;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double ref_I = 913.333;
  ASSERT_NEAR(ref_I, out_I[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, double_integral_one_var) {
  // Create data
  size_t dim = 2;
  std::vector<double> values{-17.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return 289 + f_values[1] * f_values[1]; };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double epsilon = 1e-1;
  int n = 7;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double ref_I = 6026.7;
  ASSERT_NEAR(ref_I, out_I[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, triple_integral_three_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + f_values[1] + f_values[2]; };
  std::vector<double> in_lower_limits{-4, 6, 7};
  std::vector<double> in_upper_limits{4, 13, 8};
  double epsilon = 1e-2;
  int n = 16;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double ref_I = 952;
  ASSERT_NEAR(ref_I, out_I[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, triple_integral_two_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + f_values[2]; };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double epsilon = 1e-2;
  int n = 39;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double ref_I = 9450;
  ASSERT_NEAR(ref_I, out_I[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, triple_integral_one_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, -10.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + (-10.0); };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double epsilon = 1e-2;
  int n = 65;
  std::vector<double> out_I(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
  taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
  taskDataSeq->outputs_count.emplace_back(out_I.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double ref_I = 900;
  ASSERT_NEAR(ref_I, out_I[0], epsilon);
  delete f_object;
}

// int main(int argc, char **argv) {
//   testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }////