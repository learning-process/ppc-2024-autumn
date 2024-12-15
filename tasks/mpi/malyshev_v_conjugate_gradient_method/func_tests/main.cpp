#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method_test_function {

std::vector<std::vector<double>> getRandomMatrix(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<std::vector<double>> matrix(size, std::vector<double>(size));

  for (auto &row : matrix) {
    for (auto &el : row) {
      el = dist(gen);
    }
  }

  return matrix;
}

std::vector<double> getRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> vector(size);

  for (auto &el : vector) {
    el = dist(gen);
  }

  return vector;
}

}  // namespace malyshev_v_conjugate_gradient_method_test_function

TEST(malyshev_v_conjugate_gradient_method_mpi, small_matrix_3x3) {
  uint32_t size = 3;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiSolution;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_v_conjugate_gradient_method_test_function::getRandomMatrix(size);
    randomVector = malyshev_v_conjugate_gradient_method_test_function::getRandomVector(size);
    mpiSolution.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSolution.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> seqSolution(size);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential taskSeq(taskDataSeq);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSolution.data()));
    taskDataSeq->outputs_count.push_back(size);

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < size; i++) {
      ASSERT_NEAR(seqSolution[i], mpiSolution[i], 1e-6);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, medium_matrix_100x100) {
  uint32_t size = 100;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiSolution;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_v_conjugate_gradient_method_test_function::getRandomMatrix(size);
    randomVector = malyshev_v_conjugate_gradient_method_test_function::getRandomVector(size);
    mpiSolution.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSolution.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> seqSolution(size);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential taskSeq(taskDataSeq);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSolution.data()));
    taskDataSeq->outputs_count.push_back(size);

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < size; i++) {
      ASSERT_NEAR(seqSolution[i], mpiSolution[i], 1e-6);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, large_matrix_1000x1000) {
  uint32_t size = 1000;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiSolution;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_v_conjugate_gradient_method_test_function::getRandomMatrix(size);
    randomVector = malyshev_v_conjugate_gradient_method_test_function::getRandomVector(size);
    mpiSolution.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSolution.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> seqSolution(size);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential taskSeq(taskDataSeq);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSolution.data()));
    taskDataSeq->outputs_count.push_back(size);

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < size; i++) {
      ASSERT_NEAR(seqSolution[i], mpiSolution[i], 1e-6);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, test_validation) {
  uint32_t size = 3;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiSolution;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_v_conjugate_gradient_method_test_function::getRandomMatrix(size);
    randomVector = malyshev_v_conjugate_gradient_method_test_function::getRandomVector(size);
    mpiSolution.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSolution.data()));
    taskDataPar->outputs_count.push_back(0);

    ASSERT_FALSE(taskMPI.validation());
  }
}