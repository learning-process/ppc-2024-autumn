#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/smirnov_i_tape_splitting_A/include/ops_seq.hpp"

TEST(smirnov_i_tape_splitting_A_seq, mult_matrix_and_vector) {
  int m_a = 3;
  int n_a = 3;
  int m_b = 3;
  int n_b = 1;
  auto A = new double[m_a * n_a];
  auto B = new double[m_b * n_b];
  for (int i = 0; i < n_a * m_a; i++) {
    A[i] = i;
  }
  for (int i = 0; i < n_b * m_b; i++) {
    B[i] = 2 * i;
  }
  double expected_res[] = {10., 28., 46.};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

  taskDataSeq->inputs_count.emplace_back(m_a);
  taskDataSeq->inputs_count.emplace_back(n_a);
  taskDataSeq->inputs_count.emplace_back(m_b);
  taskDataSeq->inputs_count.emplace_back(n_b);

  auto res = new double[m_a * n_b];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
  taskDataSeq->outputs_count.emplace_back(m_a);
  taskDataSeq->outputs_count.emplace_back(n_b);

  auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), true);
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();
  for (int i = 0; i < m_a * n_b; i++) {
    ASSERT_NEAR(expected_res[i], res[i], 1e-6);
  }
  delete[] A;
  delete[] B;
  delete[] res;
}
TEST(smirnov_i_tape_splitting_A_seq, mult_squered_matrix) {
  int m_a = 3;
  int n_a = 3;
  int m_b = 3;
  int n_b = 3;
  auto A = new double[m_a * n_a];
  auto B = new double[m_b * n_b];
  for (int i = 0; i < n_a * m_a; i++) {
    A[i] = i;
  }
  for (int i = 0; i < n_b * m_b; i++) {
    B[i] = 2 * i;
  }
  double expected_res[] = {30, 36, 42, 84, 108, 132, 138, 180, 222};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

  taskDataSeq->inputs_count.emplace_back(m_a);
  taskDataSeq->inputs_count.emplace_back(n_a);
  taskDataSeq->inputs_count.emplace_back(m_b);
  taskDataSeq->inputs_count.emplace_back(n_b);

  auto res = new double[m_a * n_b];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
  taskDataSeq->outputs_count.emplace_back(m_a);
  taskDataSeq->outputs_count.emplace_back(n_b);

  auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), true);
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();

  for (int i = 0; i < m_a * n_b; i++) {
    ASSERT_NEAR(expected_res[i], res[i], 1e-6);
  }
  delete[] A;
  delete[] B;
  delete[] res;
}
TEST(smirnov_i_tape_splitting_A_seq, mult_matrix_and_matrix) {
  int m_a = 2;
  int n_a = 3;
  int m_b = 3;
  int n_b = 4;
  auto A = new double[m_a * n_a];
  auto B = new double[m_b * n_b];
  for (int i = 0; i < n_a * m_a; i++) {
    A[i] = i;
  }
  for (int i = 0; i < n_b * m_b; i++) {
    B[i] = 2 * i;
  }
  double expected_res[] = {40, 46, 52, 58, 112, 136, 160, 184};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

  taskDataSeq->inputs_count.emplace_back(m_a);
  taskDataSeq->inputs_count.emplace_back(n_a);
  taskDataSeq->inputs_count.emplace_back(m_b);
  taskDataSeq->inputs_count.emplace_back(n_b);

  auto res = new double[m_a * n_b];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
  taskDataSeq->outputs_count.emplace_back(m_a);
  taskDataSeq->outputs_count.emplace_back(n_b);

  auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

  ASSERT_EQ(TestTaskSequential->validation(), true);
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();

  for (int i = 0; i < m_a * n_b; i++) {
    ASSERT_NEAR(expected_res[i], res[i], 1e-6);
  }
  delete[] A;
  delete[] B;
  delete[] res;
}
