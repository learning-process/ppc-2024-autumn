#include <gtest/gtest.h>

#include <vector>

#include "seq/vasenkov_a_gauss_jordan_method_seq/include/ops_seq.hpp"

TEST(vasenkov_a_gauss_jordan_method_seq, matrix_3x3) {
  std::vector<double> input_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
  ASSERT_TRUE(taskSequential.validation());
  ASSERT_TRUE(taskSequential.pre_processing());
  ASSERT_TRUE(taskSequential.run());
  ASSERT_TRUE(taskSequential.post_processing());

  std::vector<double> expected_result = {1, 0, 0, 250, 0, 1, 0, -130, 0, 0, 1, 20};
  ASSERT_EQ(output_result, expected_result);
}
TEST(vasenkov_a_gauss_jordan_method_seq, encorrect_data) {
  std::vector<double> input_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
  ASSERT_FALSE(taskSequential.validation());
}
TEST(vasenkov_a_gauss_jordan_method_seq, matrix_3x3_under_zerro_data) {
  std::vector<double> input_matrix = {2, -1, 1, 3, -3, -1, 2, -11, -2, 1, 2, -3};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
  ASSERT_TRUE(taskSequential.validation());
  ASSERT_TRUE(taskSequential.pre_processing());
  ASSERT_TRUE(taskSequential.run());
  ASSERT_TRUE(taskSequential.post_processing());

  std::vector<double> expected_result = {1, 0, 0, 2.8, 0, 1, 0, 2.6, 0, 0, 1, 0};
  ASSERT_EQ(output_result, expected_result);
}
TEST(vasenkov_a_gauss_jordan_method_seq, simple_matrix_3x3) {
  std::vector<double> input_matrix = {1, 0, 0, 2.8, 0, 1, 0, 2.6, 0, 0, 1, 0};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
  ASSERT_TRUE(taskSequential.validation());
  ASSERT_TRUE(taskSequential.pre_processing());
  ASSERT_TRUE(taskSequential.run());
  ASSERT_TRUE(taskSequential.post_processing());

  std::vector<double> expected_result = {1, 0, 0, 2.8, 0, 1, 0, 2.6, 0, 0, 1, 0};
  ASSERT_EQ(output_result, expected_result);
}