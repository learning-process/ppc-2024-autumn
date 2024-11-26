#include <gtest/gtest.h>

#include <vector>

#include "seq/sarafanov_m_gauss_jordan_method/include/ops_seq.hpp"

TEST(sarafanov_m_gauss_jordan_method_seq, three_simple_matrix) {
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

  sarafanov_m_gauss_jordan_method_seq::GaussJordanMethodSequential taskSequential(taskDataSeq);
  ASSERT_TRUE(taskSequential.validation());
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  std::vector<double> expected_result = {1, 0, 0, 250, 0, 1, 0, -130, 0, 0, 1, 20};
  ASSERT_EQ(output_result, expected_result);
}

TEST(sarafanov_m_gauss_jordan_method_seq, five_simple_matrix_at_1_iter) {
  std::vector<double> input_matrix = {0,  2,  3,  4, 5,  6,  0,  8,  9,  10, 11, 12, 0,  14, 15,
                                      16, 17, 18, 0, 20, 21, 22, 23, 24, 0,  26, 27, 28, 29, 30};
  int n = 5;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  sarafanov_m_gauss_jordan_method_seq::GaussJordanMethodSequential taskSequential(taskDataSeq);
  ASSERT_TRUE(taskSequential.validation());
  taskSequential.pre_processing();
  EXPECT_FALSE(taskSequential.run());
  taskSequential.post_processing();
}
