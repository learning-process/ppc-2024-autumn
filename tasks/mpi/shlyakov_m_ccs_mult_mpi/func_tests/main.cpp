#include <gtest/gtest.h>

#include "mpi/shlyakov_m_ccs_mult_mpi/include/ops_mpi.hpp"

using namespace shlyakov_m_ccs_mult_mpi;

static SparseMatrix matrix_to_ccs(const std::vector<std::vector<double>>& matrix) {
  SparseMatrix ccs_matrix;
  size_t rows = matrix.size();

  if (rows == 0) {
    ccs_matrix.col_pointers.push_back(0);
    return ccs_matrix;
  }

  size_t cols = matrix[0].size();

  ccs_matrix.col_pointers.push_back(0);

  for (size_t col = 0; col < cols; ++col) {
    for (size_t row = 0; row < rows; ++row) {
      if (matrix[row][col] != 0) {
        ccs_matrix.values.push_back(matrix[row][col]);
        ccs_matrix.row_indices.push_back(static_cast<int>(row));
      }
    }
    ccs_matrix.col_pointers.push_back(static_cast<int>(ccs_matrix.values.size()));
  }

  return ccs_matrix;
}

static std::vector<std::vector<double>> generate_random_sparse_matrix(int rows, int cols, double density) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (dis(gen) < density) {
        matrix[i][j] = normal(gen);
      }
    }
  }
  return matrix;
}

static std::vector<std::vector<double>> ccs_to_matrix(const SparseMatrix& ccs_matrix, int rows, int cols) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));

  int num_cols = static_cast<int>(ccs_matrix.col_pointers.size()) - 1;

  for (int col = 0; col < num_cols; ++col) {
    int start = ccs_matrix.col_pointers[col];
    int end = ccs_matrix.col_pointers[col + 1];
    for (int k = start; k < end; ++k) {
      int row = ccs_matrix.row_indices[k];
      matrix[row][col] = ccs_matrix.values[k];
    }
  }
  return matrix;
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<std::vector<double>> dense_A = {{1.0, 0.0, 2.0}, {0.0, 3.0, 0.0}, {4.0, 0.0, 5.0}};
  std::vector<std::vector<double>> dense_B = {{7.0, 0.0}, {0.0, 8.0}, {9.0, 0.0}};

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (rank == 0) {
    const auto* C_values = reinterpret_cast<double*>(taskData->outputs[0]);
    const auto* C_row_indices = reinterpret_cast<int*>(taskData->outputs[1]);
    const auto* C_col_pointers = reinterpret_cast<int*>(taskData->outputs[2]);

    auto C_cols = (taskData->outputs_count[2] / sizeof(int)) - 1;
    auto C_rows = static_cast<size_t>(dense_A.size());

    SparseMatrix C_ccs;
    C_ccs.values.assign(C_values, C_values + (taskData->outputs_count[0] / sizeof(double)));
    C_ccs.row_indices.assign(C_row_indices, C_row_indices + (taskData->outputs_count[1] / sizeof(int)));
    C_ccs.col_pointers.assign(C_col_pointers, C_col_pointers + static_cast<size_t>(C_cols + 1));

    std::vector<std::vector<double>> dense_C = ccs_to_matrix(C_ccs, static_cast<int>(C_rows), static_cast<int>(C_cols));
    std::vector<std::vector<double>> expected_C = {{25.0, 0.0}, {0.0, 24.0}, {73.0, 0.0}};

    ASSERT_EQ(dense_C.size(), expected_C.size());

    for (size_t i = 0; i < dense_C.size(); ++i) {
      for (size_t j = 0; j < dense_C[0].size(); ++j) {
        ASSERT_NEAR(dense_C[i][j], expected_C[i][j], 1e-6);
      }
    }
  }
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication_empty) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;

  std::vector<std::vector<double>> dense_A = {};
  std::vector<std::vector<double>> dense_B = {};

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication_singleelement) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<std::vector<double>> dense_A = {{5.0}};
  std::vector<std::vector<double>> dense_B = {{2.0}};

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (rank == 0) {
    const auto* C_values = reinterpret_cast<double*>(taskData->outputs[0]);
    const auto* C_row_indices = reinterpret_cast<int*>(taskData->outputs[1]);
    const auto* C_col_pointers = reinterpret_cast<int*>(taskData->outputs[2]);
    auto C_cols = (taskData->outputs_count[2] / sizeof(int)) - 1;
    auto C_rows = static_cast<size_t>(dense_A.size());

    SparseMatrix C_ccs;
    C_ccs.values.assign(C_values, C_values + (taskData->outputs_count[0] / sizeof(double)));
    C_ccs.row_indices.assign(C_row_indices, C_row_indices + (taskData->outputs_count[1] / sizeof(int)));
    C_ccs.col_pointers.assign(C_col_pointers, C_col_pointers + static_cast<size_t>(C_cols + 1));

    std::vector<std::vector<double>> dense_C = ccs_to_matrix(C_ccs, static_cast<int>(C_rows), static_cast<int>(C_cols));
    std::vector<std::vector<double>> expected_C = {{10.0}};

    ASSERT_EQ(dense_C.size(), expected_C.size());

    for (size_t i = 0; i < dense_C.size(); ++i) {
      for (size_t j = 0; j < dense_C[0].size(); ++j) {
        ASSERT_NEAR(dense_C[i][j], expected_C[i][j], 1e-6);
      }
    }
  }
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication_rectangular) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<std::vector<double>> dense_A = {{1.0, 2.0, 3.0}};
  std::vector<std::vector<double>> dense_B = {{4.0}, {5.0}, {6.0}};

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (rank == 0) {
    const auto* C_values = reinterpret_cast<double*>(taskData->outputs[0]);
    const auto* C_row_indices = reinterpret_cast<int*>(taskData->outputs[1]);
    const auto* C_col_pointers = reinterpret_cast<int*>(taskData->outputs[2]);

    auto C_cols = (taskData->outputs_count[2] / sizeof(int)) - 1;
    auto C_rows = static_cast<size_t>(dense_A.size());

    SparseMatrix C_ccs;
    C_ccs.values.assign(C_values, C_values + (taskData->outputs_count[0] / sizeof(double)));
    C_ccs.row_indices.assign(C_row_indices, C_row_indices + (taskData->outputs_count[1] / sizeof(int)));
    C_ccs.col_pointers.assign(C_col_pointers, C_col_pointers + static_cast<size_t>(C_cols + 1));

    std::vector<std::vector<double>> dense_C = ccs_to_matrix(C_ccs, static_cast<int>(C_rows), static_cast<int>(C_cols));
    std::vector<std::vector<double>> expected_C = {{32.0}};

    ASSERT_EQ(dense_C.size(), expected_C.size());
    for (size_t i = 0; i < dense_C.size(); ++i) {
      for (size_t j = 0; j < dense_C[0].size(); ++j) {
        ASSERT_NEAR(dense_C[i][j], expected_C[i][j], 1e-6);
      }
    }
  }
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication_zeromatrix) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<std::vector<double>> dense_A = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  std::vector<std::vector<double>> dense_B = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (rank == 0) {
    const auto* C_values = reinterpret_cast<double*>(taskData->outputs[0]);
    const auto* C_row_indices = reinterpret_cast<int*>(taskData->outputs[1]);
    const auto* C_col_pointers = reinterpret_cast<int*>(taskData->outputs[2]);

    auto C_cols = (taskData->outputs_count[2] / sizeof(int)) - 1;
    auto C_rows = static_cast<size_t>(dense_A.size());

    SparseMatrix C_ccs;
    C_ccs.values.assign(C_values, C_values + (taskData->outputs_count[0] / sizeof(double)));
    C_ccs.row_indices.assign(C_row_indices, C_row_indices + (taskData->outputs_count[1] / sizeof(int)));
    C_ccs.col_pointers.assign(C_col_pointers, C_col_pointers + static_cast<size_t>(C_cols + 1));

    std::vector<std::vector<double>> dense_C = ccs_to_matrix(C_ccs, static_cast<int>(C_rows), static_cast<int>(C_cols));
    std::vector<std::vector<double>> expected_C = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    ASSERT_EQ(dense_C.size(), expected_C.size());
    for (size_t i = 0; i < dense_C.size(); ++i) {
      for (size_t j = 0; j < dense_C[0].size(); ++j) {
        ASSERT_NEAR(dense_C[i][j], expected_C[i][j], 1e-6);
      }
    }
  }
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication_identity) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<std::vector<double>> dense_A = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  std::vector<std::vector<double>> dense_B = {{7.0, 0.0, 9.0}, {0.0, 8.0, 0.0}, {9.0, 0.0, 1.0}};

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (rank == 0) {
    const auto* C_values = reinterpret_cast<double*>(taskData->outputs[0]);
    const auto* C_row_indices = reinterpret_cast<int*>(taskData->outputs[1]);
    const auto* C_col_pointers = reinterpret_cast<int*>(taskData->outputs[2]);

    auto C_cols = (taskData->outputs_count[2] / sizeof(int)) - 1;
    auto C_rows = static_cast<size_t>(dense_A.size());

    SparseMatrix C_ccs;
    C_ccs.values.assign(C_values, C_values + (taskData->outputs_count[0] / sizeof(double)));
    C_ccs.row_indices.assign(C_row_indices, C_row_indices + (taskData->outputs_count[1] / sizeof(int)));
    C_ccs.col_pointers.assign(C_col_pointers, C_col_pointers + static_cast<size_t>(C_cols + 1));

    std::vector<std::vector<double>> dense_C = ccs_to_matrix(C_ccs, static_cast<int>(C_rows), static_cast<int>(C_cols));

    std::vector<std::vector<double>> expected_C = dense_B;

    ASSERT_EQ(dense_C.size(), expected_C.size());
    for (size_t i = 0; i < dense_C.size(); ++i) {
      for (size_t j = 0; j < dense_C[0].size(); ++j) {
        ASSERT_NEAR(dense_C[i][j], expected_C[i][j], 1e-6);
      }
    }
  }
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication_largesparse) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;

  int rows = 500;
  int cols = 500;
  double density = 0.1;

  std::vector<std::vector<double>> dense_A = generate_random_sparse_matrix(rows, cols, density);
  std::vector<std::vector<double>> dense_B = generate_random_sparse_matrix(rows, cols, density);

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(shlyakov_m_ccs_mult_mpi, matrix_multiplication_128_128) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;

  int rows = 128;
  int cols = 128;
  double density = 0.1;

  std::vector<std::vector<double>> dense_A = generate_random_sparse_matrix(rows, cols, density);
  std::vector<std::vector<double>> dense_B = generate_random_sparse_matrix(rows, cols, density);

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

    taskData->inputs_count.push_back(A_ccs.values.size());
    taskData->inputs_count.push_back(A_ccs.row_indices.size());
    taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
    taskData->inputs_count.push_back(B_ccs.values.size());
    taskData->inputs_count.push_back(B_ccs.row_indices.size());
    taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);
  }

  TestTaskMPI task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}
