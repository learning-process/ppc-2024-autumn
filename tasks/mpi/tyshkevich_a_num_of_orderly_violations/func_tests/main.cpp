#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/tyshkevich_a_num_of_orderly_violations/include/ops_mpi.hpp"

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ftest, Test_Max_10) {
  int size = 10;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel testTaskSequential(taskDataPar);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.data());

    // Create Task
    tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(result, local_count);
  }
}

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ftest, Test_Max_20) {
  int size = 20;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel testTaskSequential(taskDataPar);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.data());

    // Create Task
    tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(result, local_count);
  }
}

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ftest, Test_Max_50) {
  int size = 50;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel testTaskSequential(taskDataPar);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.data());

    // Create Task
    tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(result, local_count);
  }
}