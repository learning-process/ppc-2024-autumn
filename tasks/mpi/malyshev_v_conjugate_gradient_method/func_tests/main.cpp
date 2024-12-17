#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <random>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_conjugate_gradient_method {

std::vector<std::vector<double>> generateRandomSymmetricPositiveDefiniteMatrix(uint32_t size, double min_value,
                                                                               double max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(min_value, max_value);

  std::vector<std::vector<double>> matrix(size, std::vector<double>(size));

  for (uint32_t i = 0; i < size; i++) {
    for (uint32_t j = 0; j <= i; j++) {
      matrix[i][j] = dis(gen);
      matrix[j][i] = matrix[i][j];
    }
  }

  // Make the matrix positive definite by adding a multiple of the identity matrix
  for (uint32_t i = 0; i < size; i++) {
    matrix[i][i] += size;
  }

  return matrix;
}

std::vector<double> generateRandomVector(uint32_t size, double min_value, double max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(min_value, max_value);
  std::vector<double> vector(size);

  for (auto &el : vector) {
    el = dis(gen);
  }

  return vector;
}

}  // namespace malyshev_conjugate_gradient_method

TEST(malyshev_conjugate_gradient_method, test_small_system) {
  uint32_t size = 3;
  double min_value = 0.1;
  double max_value = 10.0;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> b;
  std::vector<double> x;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix =
        malyshev_conjugate_gradient_method::generateRandomSymmetricPositiveDefiniteMatrix(size, min_value, max_value);
    b = malyshev_conjugate_gradient_method::generateRandomVector(size, min_value, max_value);
    x.resize(size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
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

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(size);

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < size; i++) {
      ASSERT_NEAR(seqResult[i], x[i], 1e-6);
    }
  }
}

TEST(malyshev_conjugate_gradient_method, test_large_system) {
  uint32_t size = 100;
  double min_value = 0.1;
  double max_value = 10.0;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> b;
  std::vector<double> x;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix =
        malyshev_conjugate_gradient_method::generateRandomSymmetricPositiveDefiniteMatrix(size, min_value, max_value);
    b = malyshev_conjugate_gradient_method::generateRandomVector(size, min_value, max_value);
    x.resize(size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
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

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(size);

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < size; i++) {
      ASSERT_NEAR(seqResult[i], x[i], 1e-6);
    }
  }
}