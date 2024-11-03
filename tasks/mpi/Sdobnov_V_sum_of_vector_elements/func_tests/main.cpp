#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include "mpi/Sdobnov_V_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(Sdobnov_V_sum_of_vector_elements, FirstTest) { 
  boost::mpi::communicator world;

  int rows = 0;
  int columns = 0;
  int res;
  std::vector<int> input;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*> (input.data()));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(res, 0);
  }
}