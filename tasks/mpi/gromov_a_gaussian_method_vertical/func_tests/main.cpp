#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/gromov_a_gaussian_method_vertical/include/ops_mpi.hpp"

namespace gromov_a_gaussian_method_vertical_mpi {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100, 100);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen();
  }
  return vec;
}
}  // namespace gromov_a_gaussian_method_vertical_mpi

TEST(gromov_a_gaussian_method_vertical_mpi, Test_1) {
  boost::mpi::communicator world;
  int equations = 4;
  int size_coef_mat = equations * equations;
  int band_width = 3;
  std::vector<int> input_coefficient(size_coef_mat, 0);
  std::vector<int> input_rhs(equations, 0);
  std::vector<double> func_res(equations, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_rhs = gromov_a_gaussian_method_vertical_mpi::getRandomVector(equations);
    input_coefficient = gromov_a_gaussian_method_vertical_mpi::getRandomVector(size_coef_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataPar->inputs_count.emplace_back(input_coefficient.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataPar->inputs_count.emplace_back(input_rhs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel MPIGaussVerticalParallel(taskDataPar, band_width);
  ASSERT_EQ(MPIGaussVerticalParallel.validation(), true);
  MPIGaussVerticalParallel.pre_processing();
  MPIGaussVerticalParallel.run();
  MPIGaussVerticalParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(equations, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataSeq->inputs_count.emplace_back(input_rhs.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalSequential MPIGaussVerticalSequential(taskDataSeq,
                                                                                                 band_width);
    ASSERT_EQ(MPIGaussVerticalSequential.validation(), true);
    MPIGaussVerticalSequential.pre_processing();
    MPIGaussVerticalSequential.run();
    MPIGaussVerticalSequential.post_processing();

    for (int i = 0; i < equations; i++) {
      ASSERT_NEAR(func_res[i], reference_res[i], 1e-9);
    }
  }
}

TEST(gromov_a_gaussian_method_vertical_mpi, Test_2) {
  boost::mpi::communicator world;
  int equations = 2;
  int size_coef_mat = equations * equations;
  int band_width = 2;
  std::vector<int> input_coefficient(size_coef_mat, 0);
  std::vector<int> input_rhs(equations, 0);
  std::vector<double> func_res(equations, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_rhs = gromov_a_gaussian_method_vertical_mpi::getRandomVector(equations);
    input_coefficient = gromov_a_gaussian_method_vertical_mpi::getRandomVector(size_coef_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataPar->inputs_count.emplace_back(input_coefficient.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataPar->inputs_count.emplace_back(input_rhs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel MPIGaussVerticalParallel(taskDataPar, band_width);
  ASSERT_EQ(MPIGaussVerticalParallel.validation(), true);
  MPIGaussVerticalParallel.pre_processing();
  MPIGaussVerticalParallel.run();
  MPIGaussVerticalParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(equations, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataSeq->inputs_count.emplace_back(input_rhs.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalSequential MPIGaussVerticalSequential(taskDataSeq,
                                                                                                 band_width);

    ASSERT_EQ(MPIGaussVerticalSequential.validation(), true);
    MPIGaussVerticalSequential.pre_processing();
    MPIGaussVerticalSequential.run();
    MPIGaussVerticalSequential.post_processing();

    for (int i = 0; i < equations; i++) {
      ASSERT_NEAR(func_res[i], reference_res[i], 1e-9);
    }
  }
}

TEST(gromov_a_gaussian_method_vertical_mpi, Test_3) {
  boost::mpi::communicator world;
  int equations = 6;
  int size_coef_mat = equations * equations;
  int band_width = 6;
  std::vector<int> input_coefficient(size_coef_mat, 0);
  std::vector<int> input_rhs(equations, 0);
  std::vector<double> func_res(equations, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_rhs = gromov_a_gaussian_method_vertical_mpi::getRandomVector(equations);
    input_coefficient = gromov_a_gaussian_method_vertical_mpi::getRandomVector(size_coef_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataPar->inputs_count.emplace_back(input_coefficient.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataPar->inputs_count.emplace_back(input_rhs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel MPIGaussVerticalParallel(taskDataPar, band_width);
  ASSERT_EQ(MPIGaussVerticalParallel.validation(), true);
  MPIGaussVerticalParallel.pre_processing();
  MPIGaussVerticalParallel.run();
  MPIGaussVerticalParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(equations, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coefficient.data()));
    taskDataSeq->inputs_count.emplace_back(input_coefficient.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_rhs.data()));
    taskDataSeq->inputs_count.emplace_back(input_rhs.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalSequential MPIGaussVerticalSequential(taskDataSeq,
                                                                                                 band_width);

    ASSERT_EQ(MPIGaussVerticalSequential.validation(), true);
    MPIGaussVerticalSequential.pre_processing();
    MPIGaussVerticalSequential.run();
    MPIGaussVerticalSequential.post_processing();

    for (int i = 0; i < equations; i++) {
      ASSERT_NEAR(func_res[i], reference_res[i], 1e-9);
    }
  }
}