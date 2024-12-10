#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

std::vector<std::vector<double>> generate_matrix(int size) {
  std::vector<std::vector<double>> matrix(size, std::vector<double>(size, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i][j] = dist(gen);
    }
  }

  return matrix;
}

std::vector<double> generate_vector_b(int size) {
  std::vector<double> vector_b(size, 0.0);
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    vector_b[i] = dist(gen);
  }

  return vector_b;
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(Parallel_Operations_MPI, Test_2x2) {
  boost::mpi::communicator world;
  int size = 2;

  auto matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::generate_matrix(size);
  auto vector_b = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::generate_vector_b(size);

  std::vector<double> output_data(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel testMpiTaskParallel(
      taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  // Используем std::chrono для замера времени
  auto start_time = std::chrono::high_resolution_clock::now();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  if (world.rank() == 0) {
    std::cout << "Elapsed time for Test_2x2: " << elapsed_time.count() << " seconds." << std::endl;
  }

  if (world.rank() == 0) {
    // Create data for sequential processing and compare results
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(reference_data[i], output_data[i]);
    }
  }
}

TEST(Parallel_Operations_MPI, Test_5x5) {
  boost::mpi::communicator world;
  int size = 5;

  auto matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::generate_matrix(size);
  auto vector_b = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::generate_vector_b(size);

  std::vector<double> output_data(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel testMpiTaskParallel(
      taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  // Используем std::chrono для замера времени
  auto start_time = std::chrono::high_resolution_clock::now();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  if (world.rank() == 0) {
    std::cout << "Elapsed time for Test_5x5: " << elapsed_time.count() << " seconds." << std::endl;
  }

  if (world.rank() == 0) {
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(reference_data[i], output_data[i]);
    }
  }
}

TEST(Parallel_Operations_MPI, Test_invalid_data) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<std::vector<double>> matrix = {{2, 3}, {5, 4}, {1, 6}, {8, 9}};
  std::vector<double> vector_b = {1, 2, 3};

  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel testMpiTaskParallel(
        taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(Parallel_Operations_MPI, Test_not_enough_data) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<std::vector<double>> matrix = {{2, 3}, {5, 4}};
  std::vector<double> vector_b = {1};

  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel testMpiTaskParallel(
        taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}
