#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"
TEST(sedova_o_vertical_ribbon_scheme, FailValid) {
  std::vector<int> input_data = {46};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}