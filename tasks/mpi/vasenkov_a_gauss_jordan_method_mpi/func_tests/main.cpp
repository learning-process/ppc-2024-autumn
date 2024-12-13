#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/vasenkov_a_gauss_jordan_method_mpi/include/GausJordanMpi.hpp"
#include "mpi/vasenkov_a_gauss_jordan_method_mpi/include/GausJordanSeq.hpp"

TEST(vasenkov_a_gauss_jordan_method_mpi, matrix_3x3) {
  std::vector<double> input_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataMpi->inputs_count.emplace_back(input_matrix.size());

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataMpi->inputs_count.emplace_back(1);

  taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataMpi->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI taskSequential(taskDataMpi);
  ASSERT_TRUE(taskSequential.validation());
  ASSERT_TRUE(taskSequential.pre_processing());
  ASSERT_TRUE(taskSequential.run());
  ASSERT_TRUE(taskSequential.post_processing());

  std::vector<double> expected_result = {1, 0, 0, 250, 0, 1, 0, -130, 0, 0, 1, 20};
  ASSERT_EQ(output_result, expected_result);
}
TEST(vasenkov_a_gauss_jordan_method_mpi, encorrect_data) {
  std::vector<double> input_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataMpi->inputs_count.emplace_back(input_matrix.size());

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataMpi->inputs_count.emplace_back(1);

  taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataMpi->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI taskSequential(taskDataMpi);
  ASSERT_FALSE(taskSequential.validation());
}
TEST(vasenkov_a_gauss_jordan_method_mpi, matrix_3x3_under_zerro_data) {
  std::vector<double> input_matrix = {2, -1, 1, 3, -3, -1, 2, -11, -2, 1, 2, -3};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataMpi->inputs_count.emplace_back(input_matrix.size());

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataMpi->inputs_count.emplace_back(1);

  taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataMpi->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataMpi);
  ASSERT_TRUE(taskSequential.validation());
  ASSERT_TRUE(taskSequential.pre_processing());
  ASSERT_TRUE(taskSequential.run());
  ASSERT_TRUE(taskSequential.post_processing());

  std::vector<double> expected_result = {1, 0, 0, 2.8, 0, 1, 0, 2.6, 0, 0, 1, 0};
  ASSERT_EQ(output_result, expected_result);
}
TEST(vasenkov_a_gauss_jordan_method_mpi, simple_matrix_3x3) {
  std::vector<double> input_matrix = {1, 0, 0, 2.8, 0, 1, 0, 2.6, 0, 0, 1, 0};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataMpi->inputs_count.emplace_back(input_matrix.size());

  taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataMpi->inputs_count.emplace_back(1);

  taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataMpi->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI taskSequential(taskDataMpi);
  ASSERT_TRUE(taskSequential.validation());
  ASSERT_TRUE(taskSequential.pre_processing());
  ASSERT_TRUE(taskSequential.run());
  ASSERT_TRUE(taskSequential.post_processing());

  std::vector<double> expected_result = {1, 0, 0, 2.8, 0, 1, 0, 2.6, 0, 0, 1, 0};
  ASSERT_EQ(output_result, expected_result);
}

TEST(vasenkov_a_gauss_jordan_method_mpi, simple_three) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
    n = 3;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  bool parRunRes = taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    bool seqRunRes = taskSequential->run();
    taskSequential->post_processing();

    if (seqRunRes && parRunRes) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_EQ(seqRunRes, parRunRes);
    }
  }
}