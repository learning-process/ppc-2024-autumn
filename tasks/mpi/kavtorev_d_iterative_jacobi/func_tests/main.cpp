#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kavtorev_d_iterative_jacobi/include/ops_mpi.hpp"

namespace kavtorev_d_iterative_jacobi_mpi {

void generateTestData(size_t n, std::vector<double>& A_flat, std::vector<double>& F) {
  std::vector<double> x;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> offDiagDist(-10.0, 10.0);
  std::uniform_real_distribution<double> diagDist(20.0, 50.0);
  std::uniform_real_distribution<double> solDist(-100.0, 100.0);

  x.resize(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = solDist(gen);
  }

  A_flat.resize(n * n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;

    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        double value = offDiagDist(gen);
        A_flat[i * n + j] = value;
        sum += std::abs(value);
      }
    }

    A_flat[i * n + i] = sum + diagDist(gen);
  }

  F.resize(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      sum += A_flat[i * n + j] * x[j];
    }
    F[i] = sum;
  }
}

void run_test(int n, double eps = 0.001, int iterations = 1000) {
  boost::mpi::communicator world;

  std::vector<double> A;
  std::vector<double> F;
  std::vector<double> mpi_X;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    kavtorev_d_iterative_jacobi_mpi::generateTestData(n, A, F);
    mpi_X.resize(n);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
    taskDataPar->inputs_count.emplace_back(F.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mpi_X.data()));
    taskDataPar->outputs_count.emplace_back(mpi_X.size());
  }

  auto taskParallel = std::make_shared<kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI>(taskDataPar);
  if (taskParallel->validation()) {
    taskParallel->pre_processing();
    bool mpi_run_res = taskParallel->run();
    taskParallel->post_processing();

    if (mpi_run_res && world.rank() == 0) {
      std::vector<double> seq_X(n);

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
      taskDataSeq->inputs_count.emplace_back(A.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
      taskDataSeq->inputs_count.emplace_back(F.size());

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_X.data()));
      taskDataSeq->outputs_count.emplace_back(seq_X.size());

      auto taskSequential =
          std::make_shared<kavtorev_d_iterative_jacobi_mpi::IterativeJacobiSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      ASSERT_TRUE(taskSequential->run());
      taskSequential->post_processing();

      ASSERT_EQ(seq_X.size(), mpi_X.size());
      double error = 0.0;
      for (int i = 0; i < n; ++i) {
        error = std::max(error, std::abs(seq_X[i] - mpi_X[i]));
      }
      EXPECT_LT(error, 2 * eps);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

void run_val(int n, double eps, int iterations, std::vector<double> A, std::vector<double> F,
             std::vector<double> mpi_X) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (n > 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
  }

  if (eps > 0.0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);
  }

  if (iterations > 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(1);
  }

  if (world.rank() == 0) {
    if (!A.empty()) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
      taskDataPar->inputs_count.emplace_back(A.size());
    }

    if (!F.empty()) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(F.data()));
      taskDataPar->inputs_count.emplace_back(F.size());
    }

    if (!mpi_X.empty()) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mpi_X.data()));
      taskDataPar->outputs_count.emplace_back(mpi_X.size());
    }
  }

  auto taskParallel = std::make_shared<kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI>(taskDataPar);

  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(true);
  }
}

}  // namespace kavtorev_d_iterative_jacobi_mpi

TEST(kavtorev_d_iterative_jacobi_mpi, one_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(1); }

TEST(kavtorev_d_iterative_jacobi_mpi, two_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(2); }

TEST(kavtorev_d_iterative_jacobi_mpi, three_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(3); }

TEST(kavtorev_d_iterative_jacobi_mpi, four_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(4); }

TEST(kavtorev_d_iterative_jacobi_mpi, five_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(5); }

TEST(kavtorev_d_iterative_jacobi_mpi, six_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(6); }

TEST(kavtorev_d_iterative_jacobi_mpi, seven_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(7); }

TEST(kavtorev_d_iterative_jacobi_mpi, eight_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(8); }

TEST(kavtorev_d_iterative_jacobi_mpi, nine_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(9); }

TEST(kavtorev_d_iterative_jacobi_mpi, ten_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(10); }

TEST(kavtorev_d_iterative_jacobi_mpi, eleven_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(11); }

TEST(kavtorev_d_iterative_jacobi_mpi, twelve_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(12); }

TEST(kavtorev_d_iterative_jacobi_mpi, thirteen_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(13); }

TEST(kavtorev_d_iterative_jacobi_mpi, fifteen_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(15); }

TEST(kavtorev_d_iterative_jacobi_mpi, sixteen_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(16); }

TEST(kavtorev_d_iterative_jacobi_mpi, seventeen_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(17); }

TEST(kavtorev_d_iterative_jacobi_mpi, nineteen_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(19); }

TEST(kavtorev_d_iterative_jacobi_mpi, twenty_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(20); }

TEST(kavtorev_d_iterative_jacobi_mpi, twenty_one_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(21); }

TEST(kavtorev_d_iterative_jacobi_mpi, twenty_three_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(23); }

TEST(kavtorev_d_iterative_jacobi_mpi, twenty_four_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(24); }

TEST(kavtorev_d_iterative_jacobi_mpi, twenty_five_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(25); }

TEST(kavtorev_d_iterative_jacobi_mpi, twenty_seven_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(27); }

TEST(kavtorev_d_iterative_jacobi_mpi, twenty_nine_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(29); }

TEST(kavtorev_d_iterative_jacobi_mpi, thirty_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(30); }

TEST(kavtorev_d_iterative_jacobi_mpi, thirty_one_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(31); }

TEST(kavtorev_d_iterative_jacobi_mpi, fifty_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(50); }

TEST(kavtorev_d_iterative_jacobi_mpi, hundred_random_matrix) { kavtorev_d_iterative_jacobi_mpi::run_test(100); }

TEST(kavtorev_d_iterative_jacobi_mpi, null_task_data) {
  kavtorev_d_iterative_jacobi_mpi::run_val(0, 0.0, 0, {}, {}, {});
}

TEST(kavtorev_d_iterative_jacobi_mpi, insufficient_inputs) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, {}, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, null_input_pointers) {
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, {}, {}, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, null_output_pointers) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, {});
}

TEST(kavtorev_d_iterative_jacobi_mpi, invalid_n) {
  std::vector<double> A(1, 1.0);
  std::vector<double> F(1, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(-1, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, invalid_eps) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, -0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, invalid_iterations) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, -10, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, null_matrix_data) {
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, {}, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, invalid_matrix_size) {
  std::vector<double> A(3, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, invalid_vector_size) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(3, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, output_size_less_than_n) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X(1, 0.0);
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, nan_in_matrix) {
  std::vector<double> A = {1.0, NAN, 0.0, 1.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, inf_in_matrix) {
  std::vector<double> A = {1.0, INFINITY, 0.0, 1.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, nan_in_vector) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F = {1.0, NAN};
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, inf_in_vector) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F = {1.0, INFINITY};
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, zero_diagonal_element) {
  std::vector<double> A = {0.0, 1.0, 1.0, 0.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_mpi, diagonal_not_dominant) {
  std::vector<double> A = {1.0, 2.0, 2.0, 1.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_mpi::run_val(2, 0.01, 100, A, F, mpi_X);
}
