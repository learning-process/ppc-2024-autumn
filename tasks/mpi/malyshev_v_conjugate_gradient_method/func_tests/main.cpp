#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_conjugate_gradient {

std::vector<double> generateRandomVector(uint32_t size, double min_value, double max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> data(size);

  for (auto& el : data) {
    el = min_value + static_cast<double>(gen()) / static_cast<double>(std::mt19937::max()) * (max_value - min_value);
  }

  return data;
}

std::vector<std::vector<double>> generateRandomMatrix(uint32_t size, double min_value, double max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<double>> data(size, std::vector<double>(size));

  for (uint32_t i = 0; i < size; ++i) {
    for (uint32_t j = 0; j <= i; ++j) {
      data[i][j] =
          min_value + static_cast<double>(gen()) / static_cast<double>(std::mt19937::max()) * (max_value - min_value);
      if (i == j) {
        data[i][j] += size * size * max_value;
      }
      data[j][i] = data[i][j];
    }
  }

  return data;
}

}  // namespace malyshev_conjugate_gradient

TEST(malyshev_conjugate_gradient, test_small_system) {
  uint32_t size = 10;
  double min_value = 1.0;
  double max_value = 10.0;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    std::vector<double> matrixData(size * size);
    std::vector<double> vectorData(size);
    mpiResult.resize(size);

    randomMatrix = malyshev_conjugate_gradient::generateRandomMatrix(size, min_value, max_value);
    randomVector = malyshev_conjugate_gradient::generateRandomVector(size, min_value, max_value);

    for (uint32_t i = 0; i < size; ++i) {
      for (uint32_t j = 0; j < size; ++j) {
        randomMatrix[i][j] /= 1000.0;
      }
    }

    std::cerr << "Input Matrix:" << std::endl;
    for (const auto& row : randomMatrix) {
      for (const auto& val : row) {
        std::cerr << val << " ";
      }
      std::cerr << std::endl;
    }

    std::cerr << "Input Vector:" << std::endl;
    for (const auto& val : randomVector) {
      std::cerr << val << " ";
    }
    std::cerr << std::endl;

    for (uint32_t i = 0; i < size; ++i) {
      for (uint32_t j = 0; j < size; ++j) {
        matrixData[i * size + j] = randomMatrix[i][j];
      }
    }

    std::copy(randomVector.begin(), randomVector.end(), vectorData.begin());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixData.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorData.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> matrixData(size * size);
    std::vector<double> vectorData(size);

    for (uint32_t i = 0; i < size; ++i) {
      for (uint32_t j = 0; j < size; ++j) {
        matrixData[i * size + j] = randomMatrix[i][j];
      }
    }

    std::copy(randomVector.begin(), randomVector.end(), vectorData.begin());

    std::vector<double> seqResult(size);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_conjugate_gradient::TestTaskSequential taskSeq(taskDataSeq);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixData.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorData.data()));
    taskDataSeq->inputs_count.push_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(size);

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < size; i++) {
      ASSERT_FALSE(std::isnan(seqResult[i]));
      ASSERT_FALSE(std::isnan(mpiResult[i]));
      ASSERT_NEAR(seqResult[i], mpiResult[i], 1e-6);
    }
  }

  world.barrier();
}