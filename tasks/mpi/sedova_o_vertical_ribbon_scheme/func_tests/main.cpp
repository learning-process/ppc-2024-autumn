#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

TEST(sedova_o_vertical_ribbon_scheme, Test_1x1_Matrix) {
  std::vector<int> matrix = {1};
  std::vector<int> vector = {1};
  std::vector<int> expected_result = {1};
  std::vector<int> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<std::uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<std::uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, expected_result);
}
