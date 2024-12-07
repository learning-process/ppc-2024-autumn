#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "seq/budazhapova_e_matrix_multiplication/include/matrix_mult.hpp"

TEST(budazhapova_e_matrix_mult_seq, ordinary_test) {
  std::vector<int> A_matrix(12, 1);
  std::vector<int> b_vector(4, 1);
  std::vector<int> out(3, 0);
  std::vector<int> ans = {4, 4, 4};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(A_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
  taskDataSeq->inputs_count.emplace_back(b_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  budazhapova_e_matrix_mult_seq::MatrixMultSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, ans);
}

TEST(budazhapova_e_matrix_mult_seq, validation_test_1) {
  std::vector<int> A_matrix(12, 1);
  std::vector<int> b_vector(5, 1);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(A_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
  taskDataSeq->inputs_count.emplace_back(b_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  budazhapova_e_matrix_mult_seq::MatrixMultSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(budazhapova_e_matrix_mult_seq, validation_test_2) {
  std::vector<int> A_matrix(12, 1);
  std::vector<int> b_vector = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(A_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
  taskDataSeq->inputs_count.emplace_back(b_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  budazhapova_e_matrix_mult_seq::MatrixMultSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}
