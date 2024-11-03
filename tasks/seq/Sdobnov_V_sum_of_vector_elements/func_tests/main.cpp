// Copyright 2024 Sdobnov Vladimir

#include <gtest/gtest.h>

#include <vector>

#include "seq/Sdobnov_V_sum_of_vector_elements/include/ops_seq.hpp"

TEST(Sdobnov_V_sum_of_vector_elements_seq, FirtsTest) {
  int rows = 0;
  int columns = 0;
  int res;
  std::vector<int> input;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(res, 0);
}