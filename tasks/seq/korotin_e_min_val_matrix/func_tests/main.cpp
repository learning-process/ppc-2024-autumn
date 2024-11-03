// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/korotin_e_min_val_matrix/include/ops_seq.hpp"

namespace korotin_e_min_val_matrix_seq {

std::vector<double> getRandomMatrix(const unsigned rows, const unsigned columns, double scal) {
  if (rows == 0 || columns == 0) {
    throw std::invalid_argument("Can't creaate matrix with 0 rows or columns");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> matrix(rows * columns);
  for (unsigned i = 0; i < rows * columns; i++) {
    matrix[i] = gen() / scal;
  }
  return matrix;
}

}  // namespace korotin_e_min_val_matrix_seq

TEST(korotin_e_min_val_matrix_seq, test_matrix_0) {
  ASSERT_ANY_THROW(korotin_e_min_val_matrix_seq::getRandomMatrix(0, 10, 100));
  ASSERT_ANY_THROW(korotin_e_min_val_matrix_seq::getRandomMatrix(10, 0, 100));
  ASSERT_ANY_THROW(korotin_e_min_val_matrix_seq::getRandomMatrix(0, 0, 100));
}

TEST(korotin_e_min_val_matrix_seq, test_matrix_5_5) {
  const unsigned rows = 5;
  const unsigned columns = 5;
  double res;

  // Create data
  std::vector<double> matrix;
  std::vector<double> min_val(1, -5);

  matrix = korotin_e_min_val_matrix_seq::getRandomMatrix(rows, columns, 100.0);
  res = *std::min_element(matrix.begin(), matrix.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_val.data()));
  taskDataSeq->outputs_count.emplace_back(min_val.size());

  // Create Task
  korotin_e_min_val_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_DOUBLE_EQ(res, min_val[0]);
}
