#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/shkurinskaya_e_gauss_jordan/include/ops_mpi.hpp"

namespace shkurinskaya_e_gauss_jordan_mpi {

std::vector<double> generate_invertible_matrix(int size) {
  std::vector<double> matrix(size * (size + 1));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    double diag = (i * (size + 1) + i);
    for (int j = 0; j < size + 1; ++j) {
      if (i != j) {
        matrix[i * (size + 1) + j] = dist(gen);
        row_sum += std::abs(matrix[i * (size + 1) + j]);
      }
    }
    matrix[diag] = row_sum + 1;
  }

  return matrix;
}

}  // namespace shkurinskaya_e_gauss_jordan_mpi

TEST(Parallel_Operations_MPI, Test_2x2) {
  boost::mpi::communicator world;
  int size = 2;

  std::vector<double> matrix = {2, 3, 5, 4, 1, 6};

  std::vector<double> output_data(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_data(size, 0.0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    // Create Task
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
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
  int size = 2;
  std::vector<double> matrix = shkurinskaya_e_gauss_jordan_mpi::generate_invertible_matrix(size);

  std::vector<double> output_data(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_data(size, 0.0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    // Create Task
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(reference_data[i], output_data[i]);
    }
  }
}

TEST(Parallel_Operations_MPI, Test_50x50) {
  boost::mpi::communicator world;
  int size = 50;
  std::vector<double> matrix = shkurinskaya_e_gauss_jordan_mpi::generate_invertible_matrix(size);

  std::vector<double> output_data(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_data(size, 0.0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    // Create Task
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
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
  std::vector<double> matrix = {2, 3, 5, 4, 1, 6, 8};

  std::vector<double> output_data(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_data(size, 0.0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    // Create Task
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(Parallel_Operations_MPI, Test_not_enough_data) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<double> matrix = {2, 3, 5};

  std::vector<double> output_data(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_data(size, 0.0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    // Create Task
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(Parallel_Operations_MPI, Test_zero_diag) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> matrix = {0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 4, 3};

  std::vector<double> output_data(size, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_data(size, 0.0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size() / (size + 1));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    taskDataSeq->outputs_count.emplace_back(reference_data.size());

    // Create Task
    shkurinskaya_e_gauss_jordan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}
