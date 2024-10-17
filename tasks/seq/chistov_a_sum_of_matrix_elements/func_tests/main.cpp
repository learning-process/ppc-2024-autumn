#include <gtest/gtest.h>

#include "seq/chistov_a_sum_of_matrix_elements/include/ops_seq.hpp"

TEST(chistov_a_sum_of_matrix_elements, test_int_sum_sequential) {
  const int n = 3;
  const int m = 4;
  std::vector<int> global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
  std::vector<int32_t> reference_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
  taskDataSeq->outputs_count.emplace_back(reference_sum.size());

  chistov_a_sum_of_matrix_elements::TestTaskSequential<int> TestTaskSequential(taskDataSeq, n, m);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  int sum = chistov_a_sum_of_matrix_elements::classic_way(global_matrix, n, m);
  ASSERT_EQ(reference_sum[0], sum);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
