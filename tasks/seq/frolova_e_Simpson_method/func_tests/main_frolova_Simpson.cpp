// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/frolova_e_Simpson_method/include/ops_seq_frolova_Simpson.hpp"

namespace frolova_e_matrix_multiplication_seq_test {

double squaresOfX(const std::vector<double> &point) {
  double x = point[0];
  return x * x;
}

double cubeOfX(const std::vector<double> &point) {
  double x = point[0];
  return x * x * x;
}

double sumOfSquaresOfXandY(const std::vector<double> &point) {
  double x = point[0];
  double y = point[1];
  return x * x + y * y;
}

double ProductOfXAndY(const std::vector<double> &point) {
  double x = point[0];
  double y = point[1];
  return x * y;
}

double sumOfSquaresOfXandYandZ(const std::vector<double> &point) {
  double x = point[0];
  double y = point[1];
  double z = point[2];
  return x * x + y * y + z * z;
}

double ProductOfSquaresOfXandYandZ(const std::vector<double> &point) {
  double x = point[0];
  double y = point[1];
  double z = point[2];
  return x * y * z;
}
}  // namespace frolova_e_matrix_multiplication_seq_test

TEST(frolova_e_Simpson_method_seq, sumOfSquaresOfXandY_test) {
  std::vector<int> values_1 = {4, 2};
  std::vector<double> values_2 = {0.0, 2.0, 0.0, 2.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(
      taskDataSeq, frolova_e_matrix_multiplication_seq_test::sumOfSquaresOfXandY);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res[0], 10.66, 0.1);
}

TEST(frolova_e_Simpson_method_seq, ProductOfXAndY_test) {
  std::vector<int> values_1 = {4, 2};
  std::vector<double> values_2 = {1.0, 4.0, 1.0, 4.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(
      taskDataSeq, frolova_e_matrix_multiplication_seq_test::ProductOfXAndY);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res[0], 56.25, 0.1);
}

TEST(frolova_e_Simpson_method_seq, sumOfSquaresOfXandYandZ_test) {
  std::vector<int> values_1 = {4, 3};
  std::vector<double> values_2 = {0.0, 2.0, 0.0, 2.0, 0.0, 2.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(
      taskDataSeq, frolova_e_matrix_multiplication_seq_test::sumOfSquaresOfXandYandZ);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res[0], 32, 0.1);
}

TEST(frolova_e_Simpson_method_seq, ProductOfSquaresOfXandYandZ_test) {
  std::vector<int> values_1 = {8, 3};
  std::vector<double> values_2 = {0.0, 2.0, 0.0, 2.0, 0.0, 2.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(
      taskDataSeq, frolova_e_matrix_multiplication_seq_test::ProductOfSquaresOfXandYandZ);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res[0], 8, 0.1);
}

TEST(frolova_e_Simpson_method_seq, squaresOfX_test) {
  std::vector<int> values_1 = {4, 1};
  std::vector<double> values_2 = {0.0, 2.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(taskDataSeq,
                                                                 frolova_e_matrix_multiplication_seq_test::squaresOfX);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res[0], 2.66, 0.1);
}

TEST(frolova_e_Simpson_method_seq, cubeOfX_test) {
  std::vector<int> values_1 = {4, 1};
  std::vector<double> values_2 = {0.0, 2.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(taskDataSeq,
                                                                 frolova_e_matrix_multiplication_seq_test::cubeOfX);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res[0], 4, 0.1);
}

//____________ASSERT_FALSE_______________________

TEST(frolova_e_Simpson_method_seq, incorrectNumberOfIntervals) {
  std::vector<int> values_1 = {4, 1};
  std::vector<double> values_2 = {0.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(taskDataSeq,
                                                                 frolova_e_matrix_multiplication_seq_test::cubeOfX);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(frolova_e_Simpson_method_seq, NumberOfIntervalsIsNotMultipleOfTheDimension) {
  std::vector<int> values_1 = {4, 3};
  std::vector<double> values_2 = {0.0, 1.0};

  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  frolova_e_Simpson_method_seq::Simpsonmethod testTaskSequential(taskDataSeq,
                                                                 frolova_e_matrix_multiplication_seq_test::cubeOfX);

  ASSERT_EQ(testTaskSequential.validation(), false);
}