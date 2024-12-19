#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/polikanov_v_gauss_band_columns/include/ops_seq.hpp"

namespace polikanov_v_gauss_band_columns_seq {

void generateGaussianCompatibleMatrix(int size, std::vector<double>& extendedMatrix, std::vector<double>& solutions) {
  solutions.resize(size);
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
  std::vector<double> input_matrix(n * (n + 1));
  std::vector<double> global_result(n);
  std::vector<double> exp_results(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  polikanov_v_gauss_band_columns_seq::generateGaussianCompatibleMatrix(n, input_matrix, exp_results);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  auto taskSequential = std::make_shared<polikanov_v_gauss_band_columns_seq::GaussBandColumnsSequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
  taskSequential->post_processing();

  ASSERT_EQ(exp_results.size(), global_result.size());
  for (size_t i = 0; i < exp_results.size(); i++) {
    EXPECT_NEAR(global_result[i], exp_results[i], 0.01);
  }
}

}  // namespace polikanov_v_gauss_band_columns_seq

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_2) {
  polikanov_v_gauss_band_columns_seq::make_test(2);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_3) {
  polikanov_v_gauss_band_columns_seq::make_test(3);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_4) {
  polikanov_v_gauss_band_columns_seq::make_test(4);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_5) {
  polikanov_v_gauss_band_columns_seq::make_test(5);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_7) {
  polikanov_v_gauss_band_columns_seq::make_test(7);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_10) {
  polikanov_v_gauss_band_columns_seq::make_test(10);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_11) {
  polikanov_v_gauss_band_columns_seq::make_test(11);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_13) {
  polikanov_v_gauss_band_columns_seq::make_test(13);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_15) {
  polikanov_v_gauss_band_columns_seq::make_test(15);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_20) {
  polikanov_v_gauss_band_columns_seq::make_test(20);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_27) {
  polikanov_v_gauss_band_columns_seq::make_test(27);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_35) {
  polikanov_v_gauss_band_columns_seq::make_test(35);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_50) {
  polikanov_v_gauss_band_columns_seq::make_test(50);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_70) {
  polikanov_v_gauss_band_columns_seq::make_test(70);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_110) {
  polikanov_v_gauss_band_columns_seq::make_test(110);
}

TEST(polikanov_v_gauss_band_columns_seq, test_random_with_matrix_size_200) {
  polikanov_v_gauss_band_columns_seq::make_test(200);
}
