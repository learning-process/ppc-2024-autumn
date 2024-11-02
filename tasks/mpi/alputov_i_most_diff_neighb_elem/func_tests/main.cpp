// Copyright 2024 Alputov Ivan
#include "mpi/alputov_i_most_diff_neighb_elem/include/ops_mpi.hpp"
#include "mpi/alputov_i_most_diff_neighb_elem/src/ops_mpi.cpp"

/*TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_Typical_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {10, 20, 40, 80, 128, 78, -12, -15, 44, 90, 51};
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
    ASSERT_EQ(outputPair[0], 78);
    ASSERT_EQ(outputPair[1], -12);
  }
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_NegativeValues_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {-4, -6, -9, -11, -12, -13, -14, -15};
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
    //ASSERT_EQ(outputPair[0], -6);
    ASSERT_EQ(outputPair[1], -9);
  }
}*/
TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_RandomLargeVector_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 10;
    const int fixedSeed = 12345;
    std::mt19937 gen(fixedSeed);
    std::uniform_int_distribution<> dist(-1000, 1000);

    inputVector.resize(count);
    for (int i = 0; i < count; ++i) {
      inputVector[i] = dist(gen);
    }

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
TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_SingleElement_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[3] = {0, 0, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {100};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_EmptyVector_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[3] = {0, 0, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_FALSE(testMpiTaskParallel.validation());
}

/* TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_EqualElements_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0,0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
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
    //ASSERT_EQ(outputPair[0], 2);
    ASSERT_EQ(outputPair[1] - outputPair[0], 0);
  }
}

 TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_TwoElements_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[3] = {0, 0, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {5, 10};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  int elementsPerProcess = testMpiTaskParallel.getElementsPerProcess();
  if (elementsPerProcess < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(outputPair[0], 5);
      ASSERT_EQ(outputPair[1], 10);
    }
  }
}

 TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_SingleElement_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[3] = {0, 0, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {100};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_EmptyVector_MPI) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[3] = {0, 0, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask testMpiTaskParallel(taskDataPar);
  ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_Typical_Sequential) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[3] = {0, 0, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {10, 20, 40, 80, 128, 78, -12, -15, 44, 90, 51};
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataSeq->inputs_count.emplace_back(inputVector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataSeq->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask testMpiTaskSequential(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
  ASSERT_TRUE(testMpiTaskSequential.pre_processing());
  ASSERT_TRUE(testMpiTaskSequential.run());
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(outputPair[0], 78);
    ASSERT_EQ(outputPair[1], -12);
  }
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_NegativeValues_Sequential) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {-3, -6, -9, -11};
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataSeq->inputs_count.emplace_back(inputVector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataSeq->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask testMpiTaskSequential(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(outputPair[0], -3);
    ASSERT_EQ(outputPair[1], -6);
  }
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_EqualElements_Sequential) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {2, 2, 2, 2, 2};
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataSeq->inputs_count.emplace_back(inputVector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataSeq->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask testMpiTaskSequential(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(outputPair[0], 2);
    ASSERT_EQ(outputPair[1], 2);
  }
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_TwoElements_Sequential) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {5, 10};
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataSeq->inputs_count.emplace_back(inputVector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataSeq->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask testMpiTaskSequential(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(outputPair[0], 5);
    ASSERT_EQ(outputPair[1], 10);
  }
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_SingleElement_Sequential) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {100};
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataSeq->inputs_count.emplace_back(inputVector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataSeq->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask testMpiTaskSequential(taskDataSeq);
  ASSERT_FALSE(testMpiTaskSequential.validation());
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_EmptyVector_Sequential) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inputVector = {};
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataSeq->inputs_count.emplace_back(inputVector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataSeq->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask testMpiTaskSequential(taskDataSeq);
  ASSERT_FALSE(testMpiTaskSequential.validation());
}

TEST(alputov_i_most_diff_neighb_elem_mpi, Test_MaxDiff_RandomLargeVector_Sequential) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 1000000;
    const int fixedSeed = 12345;
    std::mt19937 gen(fixedSeed);
    std::uniform_int_distribution<> dist(-1000, 1000);

    inputVector.resize(count);
    for (int i = 0; i < count; ++i) {
      inputVector[i] = dist(gen);
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataSeq->inputs_count.emplace_back(inputVector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataSeq->outputs_count.emplace_back(2);
  }

  alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask testMpiTaskSequential(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
  ASSERT_TRUE(testMpiTaskSequential.pre_processing());
  ASSERT_TRUE(testMpiTaskSequential.run());
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    int index = alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(inputVector);
    ASSERT_EQ(inputVector[index], outputPair[0]);
    ASSERT_EQ(inputVector[index + 1], outputPair[1]);
  }
}*/