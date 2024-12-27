#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/borisov_s_crs_mul/include/ops_seq.hpp"

static void dense_to_crs(const std::vector<double>& dense, int M, int N, std::vector<double>& values,
                         std::vector<int>& col_index, std::vector<int>& row_ptr) {
  row_ptr.resize(M + 1, 0);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double val = dense[(i * N) + j];
      if (val != 0.0) {
        values.push_back(val);
        col_index.push_back(j);
      }
    }
    row_ptr[i + 1] = static_cast<int>(values.size());
  }
}

static void generate_dense_matrix(int M, int N, double density, std::vector<double>& dense) {
  std::mt19937_64 gen(42);
  std::uniform_real_distribution<double> dist_val(0.1, 10.0);
  std::uniform_real_distribution<double> dist_density(0.0, 1.0);

  dense.resize(M * N, 0.0);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (dist_density(gen) < density) {
        dense[(i * N) + j] = dist_val(gen);
      }
    }
  }
}

TEST(borisov_s_crs_seq_test, SuccessfulCase) {
  int M = 2;
  int N = 2;
  int K = 2;

  std::vector<double> A_dense = {1, 2, 3, 4};
  std::vector<double> A_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);

  std::vector<double> B_dense = {5, 6, 7, 8};
  std::vector<double> B_values;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::vector<double> expected_C_values = {19, 22, 43, 50};
  std::vector<int> expected_C_col_index = {0, 1, 0, 1};
  std::vector<int> expected_C_row_ptr = {0, 2, 4};

  std::vector<double> C_values(expected_C_values.size(), 0.0);
  std::vector<int> C_col_index(expected_C_col_index.size(), 0);
  std::vector<int> C_row_ptr(expected_C_row_ptr.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  C_values.assign(reinterpret_cast<double*>(taskData->outputs[0]),
                  reinterpret_cast<double*>(taskData->outputs[0]) + taskData->outputs_count[0]);
  C_col_index.assign(reinterpret_cast<int*>(taskData->outputs[1]),
                     reinterpret_cast<int*>(taskData->outputs[1]) + taskData->outputs_count[1]);
  C_row_ptr.assign(reinterpret_cast<int*>(taskData->outputs[2]),
                   reinterpret_cast<int*>(taskData->outputs[2]) + taskData->outputs_count[2]);

  EXPECT_EQ(C_values, expected_C_values);
  EXPECT_EQ(C_col_index, expected_C_col_index);
  EXPECT_EQ(C_row_ptr, expected_C_row_ptr);
}

TEST(borisov_s_crs_seq_test, InvalidInputSizes) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<double> A_values = {1.0, 2.0};
  std::vector<int> A_col_index = {0, 1};
  std::vector<int> A_row_ptr = {0, 2};

  std::vector<double> B_values = {5.0, 6.0};
  std::vector<int> B_col_index = {0, 1};
  std::vector<int> B_row_ptr = {0};

  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs.resize(3);

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(borisov_s_crs_seq_test, LargeRandomMatrix) {
  int M = 1000;
  int N = 1000;
  int K = 1000;
  double density = 0.001;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;

  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(borisov_s_crs_seq_test, LargeRandomMatrix2) {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  double density = 0.02;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;

  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(borisov_s_crs_seq_test, LargeRandomMatrix3) {
  int M = 100;
  int N = 100;
  int K = 100;
  double density = 0.5;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;

  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(borisov_s_crs_seq_test, LargeRandomMatrix4) {
  int M = 100;
  int N = 100;
  int K = 50;
  double density = 0.1;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;

  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(borisov_s_crs_seq_test, LargeRandomMatrix6) {
  int M = 512;
  int N = 512;
  int K = 512;
  double density = 0.1;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;

  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(borisov_s_crs_seq_test, LargeRandomMatrix7) {
  int M = 120;
  int N = 512;
  int K = 400;
  double density = 0.1;

  std::vector<double> A_dense;
  std::vector<double> B_dense;
  generate_dense_matrix(M, N, density, A_dense);
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;

  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  std::vector<double> C_values(M * K, 0.0);
  std::vector<int> C_col_index(M * K, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()),    reinterpret_cast<uint8_t*>(A_col_index.data()),
                      reinterpret_cast<uint8_t*>(A_row_ptr.data()),   reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col_index.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};
  taskData->inputs_count = {
      static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
      static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
      static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(C_values.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {static_cast<unsigned int>(C_values.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}
