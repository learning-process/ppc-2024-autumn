#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/borisov_s_crs_mul/include/ops_mpi.hpp"

static void prepareCRSMatrix(const std::vector<std::vector<double>>& dense_matrix, std::vector<double>& values,
                             std::vector<int>& col_index, std::vector<int>& row_ptr) {
  int nrows = static_cast<int>(dense_matrix.size());
  int ncols = dense_matrix.empty() ? 0 : static_cast<int>(dense_matrix[0].size());

  values.clear();
  col_index.clear();
  row_ptr.resize(nrows + 1, 0);

  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      if (dense_matrix[i][j] != 0.0) {
        values.push_back(dense_matrix[i][j]);
        col_index.push_back(j);
      }
    }
    row_ptr[i + 1] = static_cast<int>(values.size());
  }
}

static std::vector<std::vector<double>> generateRandomDenseMatrix(int rows, int cols, double density = 0.1) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> value_dist(-10.0, 10.0);
  std::uniform_real_distribution<> prob_dist(0.0, 1.0);

  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (prob_dist(gen) < density) {
        matrix[i][j] = value_dist(gen);
      }
    }
  }
  return matrix;
}

TEST(borisov_s_crs_mpi_test, Test_3x3_Matrices) {
  boost::mpi::communicator world;

  const std::vector<std::vector<double>> A_dense = {{1.0, 0.0, 2.0}, {0.0, 3.0, 0.0}, {4.0, 0.0, 5.0}};
  const std::vector<std::vector<double>> B_dense = {{7.0, 8.0, 9.0}, {0.0, 1.0, 0.0}, {6.0, 5.0, 4.0}};

  const std::vector<std::vector<double>> C_expected = {{19.0, 18.0, 17.0}, {0.0, 3.0, 0.0}, {58.0, 57.0, 56.0}};

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values(9, 0.0);
    std::vector<int> C_col_index(9, 0);
    std::vector<int> C_row_ptr(4, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();

  if (world.rank() == 0) {
    const auto* C_mpi_values = reinterpret_cast<const double*>(taskDataPar->outputs[0]);
    const int* C_mpi_col_index = reinterpret_cast<const int*>(taskDataPar->outputs[1]);
    const int* C_mpi_row_ptr = reinterpret_cast<const int*>(taskDataPar->outputs[2]);

    std::vector<std::vector<double>> C_dense(3, std::vector<double>(3, 0.0));
    for (int i = 0; i < 3; ++i) {
      for (int j = C_mpi_row_ptr[i]; j < C_mpi_row_ptr[i + 1]; ++j) {
        C_dense[i][C_mpi_col_index[j]] = C_mpi_values[j];
      }
    }

    ASSERT_EQ(C_dense, C_expected);
  }
}

TEST(borisov_s_crs_mpi_test, Validation_Failure_InputSizeMismatch) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.resize(5);
    taskDataPar->outputs.resize(3);
    taskDataPar->inputs_count.resize(5);
    taskDataPar->outputs_count.resize(3);
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(mpiTask.validation());
  }
}

TEST(borisov_s_crs_mpi_test, Validation_Failure_DimensionMismatch) {
  boost::mpi::communicator world;

  const std::vector<std::vector<double>> A_dense = {{1.0, 0.0}, {0.0, 2.0}};
  const std::vector<std::vector<double>> B_dense = {{1.0, 2.0}};

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(mpiTask.validation());
  }
}

TEST(borisov_s_crs_mpi_test, Large_Random_Matrices) {
  boost::mpi::communicator world;

  auto A_dense = generateRandomDenseMatrix(1000, 1000, 0.01);
  auto B_dense = generateRandomDenseMatrix(1000, 1000, 0.01);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values(1000 * 1000, 0.0);
    std::vector<int> C_col_index(1000 * 1000, 0);
    std::vector<int> C_row_ptr(1001, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();
}

TEST(borisov_s_crs_mpi_test, Large_Random_Matrices2) {
  boost::mpi::communicator world;

  auto A_dense = generateRandomDenseMatrix(1024, 1024, 0.03);
  auto B_dense = generateRandomDenseMatrix(1024, 1024, 0.03);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values(1024 * 1024, 0.0);
    std::vector<int> C_col_index(1024 * 1024, 0);
    std::vector<int> C_row_ptr(1025, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();
}

TEST(borisov_s_crs_mpi_test, Large_Random_Matrices3) {
  boost::mpi::communicator world;

  auto A_dense = generateRandomDenseMatrix(100, 100, 0.5);
  auto B_dense = generateRandomDenseMatrix(100, 100, 0.5);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values(100 * 100, 0.0);
    std::vector<int> C_col_index(100 * 100, 0);
    std::vector<int> C_row_ptr(101, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();
}

TEST(borisov_s_crs_mpi_test, Large_Random_Matrices4) {
  boost::mpi::communicator world;

  auto A_dense = generateRandomDenseMatrix(2000, 2000, 0.1);
  auto B_dense = generateRandomDenseMatrix(2000, 2000, 0.1);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values(2000 * 2000, 0.0);
    std::vector<int> C_col_index(2000 * 2000, 0);
    std::vector<int> C_row_ptr(2001, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();
}

TEST(borisov_s_crs_mpi_test, Large_Random_Matrices5) {
  boost::mpi::communicator world;

  auto A_dense = generateRandomDenseMatrix(512, 512, 0.2);
  auto B_dense = generateRandomDenseMatrix(512, 512, 0.2);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values(512 * 512, 0.0);
    std::vector<int> C_col_index(512 * 512, 0);
    std::vector<int> C_row_ptr(513, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();
}

TEST(borisov_s_crs_mpi_test, Large_Random_Matrices6) {
  boost::mpi::communicator world;

  auto A_dense = generateRandomDenseMatrix(120, 512, 0.2);
  auto B_dense = generateRandomDenseMatrix(512, 400, 0.2);

  std::vector<double> A_values;
  std::vector<double> B_values;
  std::vector<int> A_col_index;
  std::vector<int> B_col_index;
  std::vector<int> A_row_ptr;
  std::vector<int> B_row_ptr;

  prepareCRSMatrix(A_dense, A_values, A_col_index, A_row_ptr);
  prepareCRSMatrix(B_dense, B_values, B_col_index, B_row_ptr);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ptr.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_index.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ptr.data()));

    taskDataPar->inputs_count = {
        static_cast<unsigned int>(A_values.size()),    static_cast<unsigned int>(A_col_index.size()),
        static_cast<unsigned int>(A_row_ptr.size()),   static_cast<unsigned int>(B_values.size()),
        static_cast<unsigned int>(B_col_index.size()), static_cast<unsigned int>(B_row_ptr.size())};

    std::vector<double> C_values(120 * 400, 0.0);
    std::vector<int> C_col_index(120 * 400, 0);
    std::vector<int> C_row_ptr(401, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_values.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_index.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ptr.data()));

    taskDataPar->outputs_count = {static_cast<unsigned int>(C_values.size()),
                                  static_cast<unsigned int>(C_col_index.size()),
                                  static_cast<unsigned int>(C_row_ptr.size())};
  }

  borisov_s_crs_mul_mpi::CrsMatrixMulTaskMPI mpiTask(taskDataPar);
  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();
}
