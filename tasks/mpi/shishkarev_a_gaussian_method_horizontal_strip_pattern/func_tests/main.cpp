#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

std::vector<double> getRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> mat(sz);
  for (int i = 0; i < sz; ++i) {
    mat[i] = dis(gen);
  }
  return mat;
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_empty_matrix) {
  boost::mpi::communicator world;

  const int cols = 0;
  const int rows = 0;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.validation());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_matrix_with_one_element) {
  boost::mpi::communicator world;

  const int cols = 1;
  const int rows = 1;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.validation());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_not_square_matrix) {
  boost::mpi::communicator world;

  const int cols = 5;
  const int rows = 2;

  std::vector<double> global_matrix;
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.validation());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_zero_determinant) {
  boost::mpi::communicator world;

  const int cols = 4;
  const int rows = 3;

  std::vector<double> global_matrix;
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {6, -1, 12, 3, -3, -5, -6, 9, 1, 4, 2, -1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.validation());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_101x100) {
  boost::mpi::communicator world;

  const int cols = 101;
  const int rows = 100;

  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::getRandomMatrix(cols * rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
      taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel.validation(), true);
  MPIGaussHorizontalParallel.pre_processing();
  MPIGaussHorizontalParallel.run();
  MPIGaussHorizontalParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(cols - 1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential
        MPIGaussHorizontalSequential(taskDataSeq);
    ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
    MPIGaussHorizontalSequential.pre_processing();
    MPIGaussHorizontalSequential.run();
    MPIGaussHorizontalSequential.post_processing();

    for (int i = 0; i < cols - 1; ++i) {
      ASSERT_NEAR(global_res[i], reference_res[i], 1e-6);
    }
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_201x200) {
  boost::mpi::communicator world;

  const int cols = 201;
  const int rows = 200;

  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::getRandomMatrix(cols * rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
      taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel.validation(), true);
  MPIGaussHorizontalParallel.pre_processing();
  MPIGaussHorizontalParallel.run();
  MPIGaussHorizontalParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(cols - 1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential
        MPIGaussHorizontalSequential(taskDataSeq);
    ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
    MPIGaussHorizontalSequential.pre_processing();
    MPIGaussHorizontalSequential.run();
    MPIGaussHorizontalSequential.post_processing();

    for (int i = 0; i < cols - 1; ++i) {
      ASSERT_NEAR(global_res[i], reference_res[i], 1e-6);
    }
  }
}
