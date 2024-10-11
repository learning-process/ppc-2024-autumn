// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/seq_titov_s_vector_sum/include/ops_seq.hpp"

TEST(titov_s_vector_sum_seq, Test_Int) {
  // Create data
  std::vector<int> in(1, 10);
  const int expected_sum = 10;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  titov_s_vector_sum_seq::VectorSumSequential<int> vectorSumSequential(taskDataSeq);
  ASSERT_EQ(vectorSumSequential.validation(), true);
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(expected_sum, out[0]);
}
