#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <random>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_conjugate_gradient_method {

std::vector<std::vector<double>> generateRandomSymmetricPositiveDefiniteMatrix(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(1.0, 10.0);

  std::vector<std::vector<double>> matrix(size, std::vector<double>(size));

  // Generate a random symmetric matrix
  for (uint32_t i = 0; i < size; ++i) {
    for (uint32_t j = 0; j <= i; ++j) {
      matrix[i][j] = matrix[j][i] = dis(gen);
    }
  }

  // Make it positive definite by adding a multiple of the identity matrix
  for (uint32_t i = 0; i < size; ++i) {
    matrix[i][i] += size;
  }

  return matrix;
}

std::vector<double> generateRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(1.0, 5.0);

  std::vector<double> vector(size);

  for (auto &el : vector) {
    el = dis(gen);
  }

  return vector;
}

}  // namespace malyshev_conjugate_gradient_method

TEST(malyshev_conjugate_gradient_method, test_small_system) {
  uint32_t size = 3;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_conjugate_gradient_method::generateRandomSymmetricPositiveDefiniteMatrix(size);
    randomVector = malyshev_conjugate_gradient_method::generateRandomVector(size);
    mpiResult.resize(size);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> seqResult(size);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_conjugate_gradient_method::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_NEAR(seqResult[i], mpiResult[i], 1e-1);
    }
  }
}

TEST(malyshev_conjugate_gradient_method, test_large_system) {
  uint32_t size = 100;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_conjugate_gradient_method::generateRandomSymmetricPositiveDefiniteMatrix(size);
    randomVector = malyshev_conjugate_gradient_method::generateRandomVector(size);
    mpiResult.resize(size);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> seqResult(size);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_conjugate_gradient_method::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_NEAR(seqResult[i], mpiResult[i], 1e-1);
    }
  }
}