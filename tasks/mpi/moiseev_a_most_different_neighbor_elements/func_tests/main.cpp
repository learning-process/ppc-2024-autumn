#include <gtest/gtest.h>

#include "mpi/moiseev_a_most_different_neighbor_elements/include/ops_mpi.hpp"
#include "seq/moiseev_a_most_different_neighbor_elements/include/ops_seq.hpp"

template <typename DataType>
std::vector<DataType> generateRandomVector(int size) {
  std::vector<DataType> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = static_cast<DataType>(rand() % 100);
  }
  return vec;
}

TEST(moiseev_a_most_different_neighbor_elements_mpi_test, TestVectorInt100) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  std::vector<int> global_vec;
  std::vector<int32_t> global_values_out(2);
  std::vector<uint64_t> global_indices_out(2);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    const int vector_size = 100;
    global_vec = generateRandomVector<int>(vector_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_values_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices_out.data()));

    taskDataPar->outputs_count.emplace_back(global_values_out.size());
    taskDataPar->outputs_count.emplace_back(global_indices_out.size());
  }

  moiseev_a_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  taskParallel.pre_processing();
  taskParallel.run();
  taskParallel.post_processing();

  if (rank == 0) {
    std::vector<int32_t> reference_values_out(2);
    std::vector<uint64_t> reference_indices_out(2);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_values_out.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_values_out.size());
    taskDataSeq->outputs_count.emplace_back(reference_indices_out.size());

    moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int> taskSequential(
        taskDataSeq);
    ASSERT_TRUE(taskSequential.validation());
    taskSequential.pre_processing();
    taskSequential.run();
    taskSequential.post_processing();

    ASSERT_EQ(reference_values_out[0], global_values_out[0]);
    ASSERT_EQ(reference_values_out[1], global_values_out[1]);
    ASSERT_EQ(reference_indices_out[0], global_indices_out[0]);
    ASSERT_EQ(reference_indices_out[1], global_indices_out[1]);
  }
}

TEST(moiseev_a_most_different_neighbor_elements_mpi_test, TestVectorDouble100) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  std::vector<double> global_vec;
  std::vector<int32_t> global_values_out(2);
  std::vector<uint64_t> global_indices_out(2);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    const int vector_size = 100;
    global_vec = generateRandomVector<double>(vector_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_values_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices_out.data()));

    taskDataPar->outputs_count.emplace_back(global_values_out.size());
    taskDataPar->outputs_count.emplace_back(global_indices_out.size());
  }

  moiseev_a_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  taskParallel.pre_processing();
  taskParallel.run();
  taskParallel.post_processing();

  if (rank == 0) {
    std::vector<int32_t> reference_values_out(2);
    std::vector<uint64_t> reference_indices_out(2);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_values_out.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_values_out.size());
    taskDataSeq->outputs_count.emplace_back(reference_indices_out.size());

    moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int> taskSequential(
        taskDataSeq);
    ASSERT_TRUE(taskSequential.validation());
    taskSequential.pre_processing();
    taskSequential.run();
    taskSequential.post_processing();

    ASSERT_EQ(reference_values_out[0], global_values_out[0]);
    ASSERT_EQ(reference_values_out[1], global_values_out[1]);
    ASSERT_EQ(reference_indices_out[0], global_indices_out[0]);
    ASSERT_EQ(reference_indices_out[1], global_indices_out[1]);
  }
}

TEST(moiseev_a_most_different_neighbor_elements_mpi_test, TestVectorFloat100) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  std::vector<float> global_vec;
  std::vector<int32_t> global_values_out(2);
  std::vector<uint64_t> global_indices_out(2);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    const int vector_size = 100;
    global_vec = generateRandomVector<float>(vector_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_values_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices_out.data()));

    taskDataPar->outputs_count.emplace_back(global_values_out.size());
    taskDataPar->outputs_count.emplace_back(global_indices_out.size());
  }

  moiseev_a_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  taskParallel.pre_processing();
  taskParallel.run();
  taskParallel.post_processing();

  if (rank == 0) {
    std::vector<int32_t> reference_values_out(2);
    std::vector<uint64_t> reference_indices_out(2);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_values_out.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_values_out.size());
    taskDataSeq->outputs_count.emplace_back(reference_indices_out.size());

    moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int> taskSequential(
        taskDataSeq);
    ASSERT_TRUE(taskSequential.validation());
    taskSequential.pre_processing();
    taskSequential.run();
    taskSequential.post_processing();

    ASSERT_EQ(reference_values_out[0], global_values_out[0]);
    ASSERT_EQ(reference_values_out[1], global_values_out[1]);
    ASSERT_EQ(reference_indices_out[0], global_indices_out[0]);
    ASSERT_EQ(reference_indices_out[1], global_indices_out[1]);
  }
}

TEST(moiseev_a_most_different_neighbor_elements_mpi_test, TestVectorWithEqualElements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(100, 5);
  std::vector<int32_t> global_values_out(2);
  std::vector<uint64_t> global_indices_out(2);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_values_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices_out.data()));
    taskDataPar->outputs_count.emplace_back(global_values_out.size());
    taskDataPar->outputs_count.emplace_back(global_indices_out.size());
  }

  moiseev_a_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  taskParallel.pre_processing();
  taskParallel.run();
  taskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_values_out[0], global_values_out[1]);
  }
}

TEST(moiseev_a_most_different_neighbor_elements_mpi_test, TestSmallVector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {3, 7};
  std::vector<int32_t> global_values_out(2);
  std::vector<uint64_t> global_indices_out(2);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_values_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices_out.data()));
    taskDataPar->outputs_count.emplace_back(global_values_out.size());
    taskDataPar->outputs_count.emplace_back(global_indices_out.size());
  }

  moiseev_a_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  taskParallel.pre_processing();
  taskParallel.run();
  taskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(4, std::abs(global_vec[1] - global_vec[0]));
    ASSERT_EQ(global_indices_out[0], 0u);
    ASSERT_EQ(global_indices_out[1], 1u);
  }
}

TEST(moiseev_a_most_different_neighbor_elements_mpi_test, TestLargeRandomVector) {
  boost::mpi::communicator world;
  const int vector_size = 10000;
  auto global_vec = generateRandomVector<int>(vector_size);
  std::vector<int32_t> global_values_out(2);
  std::vector<uint64_t> global_indices_out(2);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_values_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices_out.data()));
    taskDataPar->outputs_count.emplace_back(global_values_out.size());
    taskDataPar->outputs_count.emplace_back(global_indices_out.size());
  }

  moiseev_a_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  taskParallel.pre_processing();
  taskParallel.run();
  taskParallel.post_processing();
}

TEST(moiseev_a_most_different_neighbor_elements_mpi_test, TestVectorWithNegativeValues) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {-5, -10, 0, 3, -2};
  std::vector<int32_t> global_value_out(2);
  std::vector<uint64_t> global_indices_out(2);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_value_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices_out.data()));
    taskDataPar->outputs_count.emplace_back(global_value_out.size());
    taskDataPar->outputs_count.emplace_back(global_indices_out.size());
  }

  moiseev_a_most_different_neighbor_elements_mpi::MostDifferentNeighborElementsParallel<int> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  taskParallel.pre_processing();
  taskParallel.run();
  taskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_value_out[0], -10);
    ASSERT_EQ(global_value_out[1], 0);
  }
}