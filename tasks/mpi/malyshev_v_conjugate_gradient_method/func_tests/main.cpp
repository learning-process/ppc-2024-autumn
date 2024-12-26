#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method {

std::vector<std::vector<double>> getSmallMatrix() { return {{4, 1}, {1, 3}}; }

std::vector<double> getSmallVector() { return {1, 2}; }

std::vector<std::vector<double>> getLargeMatrix() { return {{5, 1, 0, 0}, {1, 5, 1, 0}, {0, 1, 5, 1}, {0, 0, 1, 5}}; }

std::vector<double> getLargeVector() { return {1, 2, 3, 4}; }

}  // namespace malyshev_v_conjugate_gradient_method

TEST(malyshev_v_conjugate_gradient_method, test_small_system) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix = malyshev_v_conjugate_gradient_method::getSmallMatrix();
    vector = malyshev_v_conjugate_gradient_method::getSmallVector();
    mpiResult.resize(vector.size());

    for (auto &row : matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(mpiResult.size());
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> expectedResult = {0.090909, 0.636364};
    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_NEAR(expectedResult[i], mpiResult[i], 1e-5);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method, test_large_system) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix = malyshev_v_conjugate_gradient_method::getLargeMatrix();
    vector = malyshev_v_conjugate_gradient_method::getLargeVector();
    mpiResult.resize(vector.size());

    for (auto &row : matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(mpiResult.size());
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<double> expectedResult = {0.090909, 0.181818, 0.272727,
                                          0.363636};
    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_NEAR(expectedResult[i], mpiResult[i], 1e-5);
    }
  }
}