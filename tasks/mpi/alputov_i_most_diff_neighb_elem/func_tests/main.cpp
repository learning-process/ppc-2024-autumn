// Copyright 2024 Alputov Ivan
#include "mpi/alputov_i_most_diff_neighb_elem/include/ops_mpi.hpp"
#include "mpi/alputov_i_most_diff_neighb_elem/src/ops_mpi.cpp"

TEST(alputov_i_most_diff_neighb_elem_mpi, Test__MPI_10_elem_check_validation) {
  boost::mpi::communicator world;
  std::vector<int> inputVector(10);
  int outputPair[2] = {};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = alputov_i_most_diff_neighb_elem_mpi::RandomVector(10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  int proc = testMpiTaskParallel.getElementsPerProcess();
  if (proc >= 2)
    ASSERT_TRUE(testMpiTaskParallel.validation());
  else
    ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_RandomLargeVector_MPI_1000_elem) {
  boost::mpi::communicator world;
  std::vector<int> inputVector(1000);
  int outputPair[2] = {};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = alputov_i_most_diff_neighb_elem_mpi::RandomVector(1000);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int index = alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(inputVector);
    ASSERT_EQ(inputVector[index], outputPair[0]);
    ASSERT_EQ(inputVector[index + 1], outputPair[1]);
  }
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_RandomVector_MPI_100_elem) {
  boost::mpi::communicator world;
  std::vector<int> inputVector(100);
  int outputPair[2] = {};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = alputov_i_most_diff_neighb_elem_mpi::RandomVector(100);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int index = alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(inputVector);
    ASSERT_EQ(inputVector[index], outputPair[0]);
    ASSERT_EQ(inputVector[index + 1], outputPair[1]);
  }
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_LargeVector_MPI_with_1000_equal_elem) {
  boost::mpi::communicator world;
  std::vector<int> inputVector(1000);
  int outputPair[2] = {};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::fill(inputVector.begin(), inputVector.end(), 2);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int index = alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(inputVector);
    ASSERT_EQ(inputVector[index], outputPair[0]);
    ASSERT_EQ(inputVector[index + 1], outputPair[1]);
  }
}
TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_RandomLargeVector_MPI_1_million_elem) {
  boost::mpi::communicator world;
  std::vector<int> inputVector(1000000);
  int outputPair[2] = {};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = alputov_i_most_diff_neighb_elem_mpi::RandomVector(1000000);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int index = alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(inputVector);
    ASSERT_EQ(inputVector[index], outputPair[0]);
    ASSERT_EQ(inputVector[index + 1], outputPair[1]);
  }
}
