#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_mpi.hpp"

namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi {

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

}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi

void runMatrixMultiplication(int rowsA, int columnsA, int columnsB) {
  boost::mpi::communicator _world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_parallel;
  std::vector<int> global_result_sequential;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (_world.rank() == 0) {
    global_matrix_a = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(rowsA, columnsA);
    global_matrix_b = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(columnsA, columnsB);
    global_result_parallel.resize(rowsA * columnsB, 0);
    global_result_sequential.resize(rowsA * columnsB, 0);

    auto add_task_data = [&global_matrix_a, &global_matrix_b, &rowsA, &columnsA, &columnsB](
                             std::shared_ptr<ppc::core::TaskData>& taskData, std::vector<int>& result) {
      std::vector<std::pair<void*, size_t>> inputs = {{global_matrix_a.data(), global_matrix_a.size()},
                                                      {global_matrix_b.data(), global_matrix_b.size()},
                                                      {&rowsA, 1},
                                                      {&columnsA, 1},
                                                      {&columnsB, 1}};

      for (const auto& [ptr, size] : inputs) {
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(ptr));
        taskData->inputs_count.emplace_back(size);
      }

      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
      taskData->outputs_count.emplace_back(result.size());
    };

    add_task_data(taskDataPar, global_result_parallel);
    add_task_data(taskDataSeq, global_result_sequential);
  }

  auto taskParallel =
      std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (_world.rank() == 0) {
    auto taskSequential =
        std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::SequentialMatrixMultiplicationTask>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();
    ASSERT_EQ(global_result_parallel, global_result_sequential);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_zero) {
  boost::mpi::communicator _world;

  int rowA;
  int colA;
  int colB;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result_parallel;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (_world.rank() == 0) {
    rowA = 1;
    colA = 0;
    colB = 1;

    global_matrix_a = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(rowA, colA);
    global_matrix_b = kalinin_d_matrix_mult_hor_a_vert_b_mpi::getRandomMatrix(colA, colB);
    global_result_parallel.resize(rowA * colB, 0);

    auto add_task_data = [&global_matrix_a, &global_matrix_b, &rowA, &colA, &colB](
                             std::shared_ptr<ppc::core::TaskData>& taskData, std::vector<int>& result) {
      std::vector<std::pair<void*, size_t>> inputs = {{global_matrix_a.data(), global_matrix_a.size()},
                                                      {global_matrix_b.data(), global_matrix_b.size()},
                                                      {&rowA, 1},
                                                      {&colA, 1},
                                                      {&colB, 1}};

      for (const auto& [ptr, size] : inputs) {
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(ptr));
        taskData->inputs_count.emplace_back(size);
      }

      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
      taskData->outputs_count.emplace_back(result.size());
    };

    add_task_data(taskDataPar, global_result_parallel);
  }

  auto taskParallel =
      std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask>(taskDataPar);

  if (_world.rank() == 0) {
    ASSERT_FALSE(taskParallel->validation());
  } else {
    ASSERT_TRUE(taskParallel->validation());
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_1x1) { runMatrixMultiplication(1, 1, 1); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_2x2) { runMatrixMultiplication(2, 2, 2); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_3x3) { runMatrixMultiplication(3, 3, 3); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_10x10) { runMatrixMultiplication(10, 10, 10); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_100x100) { runMatrixMultiplication(100, 100, 100); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_3x2_2x3) { runMatrixMultiplication(3, 2, 3); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_5x4_4x5) { runMatrixMultiplication(5, 4, 5); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_1x10_10x16) { runMatrixMultiplication(1, 10, 16); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_100x3_3x100) { runMatrixMultiplication(100, 3, 100); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_10x15_15x18) { runMatrixMultiplication(10, 15, 18); }

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, matrix_40x80_80x100) { runMatrixMultiplication(40, 80, 100); }
