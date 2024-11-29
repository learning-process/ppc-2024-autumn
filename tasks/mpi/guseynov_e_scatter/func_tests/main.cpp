#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/guseynov_e_scatter/include/ops_mpi.hpp"

TEST(guseynov_e_scatter, Test_non_random_aray_10){
    boost::mpi::communicator world;
    std::vector<int> global_vec = {1, 2,  3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int32_t> global_res(1, -1);
     // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_scatter::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    // create Task
    guseynov_e_scatter::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}