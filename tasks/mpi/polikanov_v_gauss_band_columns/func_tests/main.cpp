#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/polikanov_v_gauss_band_columns/include/ops_mpi.hpp"

namespace polikanov_v_gauss_band_columns_mpi {

void generateGaussianCompatibleMatrix(int size, std::vector<double>& extendedMatrix) {
  std::vector<double> solutions(size);
  extendedMatrix.resize(size * (size + 1));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-100.0, 100.0);

  for (int i = 0; i < size; ++i) {
    solutions[i] = dist(gen);
  }

  for (int i = 0; i < size; ++i) {
    double sum = 0.0;
    for (int j = 0; j < size; ++j) {
      double value = dist(gen);
      extendedMatrix[i * (size + 1) + j] = value;
      sum += value * solutions[j];
    }
    extendedMatrix[i * (size + 1) + size] = sum;
  }
}

void make_test(size_t n) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix(n * (n + 1));
  std::vector<double> global_result(n);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    polikanov_v_gauss_band_columns_mpi::generateGaussianCompatibleMatrix(n, input_matrix);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<polikanov_v_gauss_band_columns_mpi::GaussBandColumnsParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_results(global_result.size());

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_results.data()));
    taskDataSeq->outputs_count.emplace_back(seq_results.size());

    auto taskSequential =
        std::make_shared<polikanov_v_gauss_band_columns_mpi::GaussBandColumnsSequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(seq_results.size(), global_result.size());
    for (size_t i = 0; i < seq_results.size(); i++) {
      EXPECT_NEAR(global_result[i], seq_results[i], 0.01);
    }
  }
}

}  // namespace polikanov_v_gauss_band_columns_mpi

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_2) {
  polikanov_v_gauss_band_columns_mpi::make_test(2);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_3) {
  polikanov_v_gauss_band_columns_mpi::make_test(3);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_4) {
  polikanov_v_gauss_band_columns_mpi::make_test(4);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_5) {
  polikanov_v_gauss_band_columns_mpi::make_test(5);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_7) {
  polikanov_v_gauss_band_columns_mpi::make_test(7);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_10) {
  polikanov_v_gauss_band_columns_mpi::make_test(10);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_11) {
  polikanov_v_gauss_band_columns_mpi::make_test(11);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_13) {
  polikanov_v_gauss_band_columns_mpi::make_test(13);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_15) {
  polikanov_v_gauss_band_columns_mpi::make_test(15);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_20) {
  polikanov_v_gauss_band_columns_mpi::make_test(20);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_27) {
  polikanov_v_gauss_band_columns_mpi::make_test(27);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_35) {
  polikanov_v_gauss_band_columns_mpi::make_test(35);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_50) {
  polikanov_v_gauss_band_columns_mpi::make_test(50);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_70) {
  polikanov_v_gauss_band_columns_mpi::make_test(70);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_110) {
  polikanov_v_gauss_band_columns_mpi::make_test(110);
}

TEST(polikanov_v_gauss_band_columns_mpi, test_random_with_matrix_size_200) {
  polikanov_v_gauss_band_columns_mpi::make_test(200);
}
