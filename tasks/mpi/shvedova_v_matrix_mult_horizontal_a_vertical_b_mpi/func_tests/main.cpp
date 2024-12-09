#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi/include/ops_mpi.hpp"

namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi {

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-1000, 1000);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi

void RunMatrixMultiplicationTest(int rowA, int colA, int colB) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_parallel;
  std::vector<int> global_result_sequential;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix_a = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(rowA, colA);
    global_matrix_b = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(colA, colB);
    global_result_parallel.resize(rowA * colB, 0);
    global_result_sequential.resize(rowA * colB, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(global_result_parallel.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix_a.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix_b.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(global_result_sequential.size());
  }

  auto taskParallel =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel>(
          taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    auto taskSequential =
        std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskSequential>(
            taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_result_parallel, global_result_sequential);
  }
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, square_1x1) { RunMatrixMultiplicationTest(2, 2, 2); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, square_2x2) { RunMatrixMultiplicationTest(2, 2, 2); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, square_3x3) { RunMatrixMultiplicationTest(3, 3, 3); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, square_5x5) { RunMatrixMultiplicationTest(3, 3, 3); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, rec_4x1_1x5) { RunMatrixMultiplicationTest(4, 3, 5); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, rec_4x3_3x5) { RunMatrixMultiplicationTest(4, 3, 5); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, rec_5x4_4x5) { RunMatrixMultiplicationTest(4, 3, 5); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, sq_1x1_1x1) { RunMatrixMultiplicationTest(1, 1, 1); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, rec_7x11_11x13) { RunMatrixMultiplicationTest(7, 11, 13); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, rec_4x8_8x16) { RunMatrixMultiplicationTest(4, 8, 16); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, rec_1x7_7x16) { RunMatrixMultiplicationTest(1, 7, 16); }

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi, validation_zero_matrix) {
  boost::mpi::communicator world;

  int rowA;
  int colA;
  int colB;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_parallel;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    rowA = 1;
    colA = 0;
    colB = 1;

    global_matrix_a = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(rowA, colA);
    global_matrix_b = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(colA, colB);
    global_result_parallel.resize(rowA * colB, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(global_result_parallel.size());
  }

  auto taskParallel =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel>(
          taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(taskParallel->validation());
  } else {
    ASSERT_TRUE(taskParallel->validation());
  }
}
