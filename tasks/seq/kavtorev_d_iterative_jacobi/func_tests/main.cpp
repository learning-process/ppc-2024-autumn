#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/kavtorev_d_iterative_jacobi/include/ops_seq.hpp"

namespace kavtorev_d_iterative_jacobi_seq {

void generateTestData(size_t n, std::vector<double>& A_flat, std::vector<double>& F, std::vector<double>& x) {
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
  std::vector<double> A;
  std::vector<double> F;
  std::vector<double> exp_X;
  kavtorev_d_iterative_jacobi_seq::generateTestData(n, A, F, exp_X);
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

  auto taskSequential = std::make_shared<kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  bool seq_run_res = taskSequential->run();
  taskSequential->post_processing();

  if (seq_run_res) {
    ASSERT_EQ(seq_X.size(), exp_X.size());
    double error = 0.0;
    for (int i = 0; i < n; ++i) {
      error = std::max(error, std::abs(exp_X[i] - seq_X[i]));
    }
    EXPECT_LT(error, eps);
  }
}

void run_val(int n, double eps, int iterations, std::vector<double> A, std::vector<double> F,
             std::vector<double> mpi_X) {
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

  auto taskParallel = std::make_shared<kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential>(taskDataPar);

  EXPECT_FALSE(taskParallel->validation());
}

}  // namespace kavtorev_d_iterative_jacobi_seq

TEST(kavtorev_d_iterative_jacobi_seq, one_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(1); }

TEST(kavtorev_d_iterative_jacobi_seq, two_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(2); }

TEST(kavtorev_d_iterative_jacobi_seq, three_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(3); }

TEST(kavtorev_d_iterative_jacobi_seq, four_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(4); }

TEST(kavtorev_d_iterative_jacobi_seq, five_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(5); }

TEST(kavtorev_d_iterative_jacobi_seq, six_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(6); }

TEST(kavtorev_d_iterative_jacobi_seq, seven_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(7); }

TEST(kavtorev_d_iterative_jacobi_seq, eight_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(8); }

TEST(kavtorev_d_iterative_jacobi_seq, nine_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(9); }

TEST(kavtorev_d_iterative_jacobi_seq, ten_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(10); }

TEST(kavtorev_d_iterative_jacobi_seq, eleven_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(11); }

TEST(kavtorev_d_iterative_jacobi_seq, twelve_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(12); }

TEST(kavtorev_d_iterative_jacobi_seq, thirteen_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(13); }

TEST(kavtorev_d_iterative_jacobi_seq, fifteen_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(15); }

TEST(kavtorev_d_iterative_jacobi_seq, sixteen_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(16); }

TEST(kavtorev_d_iterative_jacobi_seq, seventeen_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(17); }

TEST(kavtorev_d_iterative_jacobi_seq, nineteen_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(19); }

TEST(kavtorev_d_iterative_jacobi_seq, twenty_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(20); }

TEST(kavtorev_d_iterative_jacobi_seq, twenty_one_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(21); }

TEST(kavtorev_d_iterative_jacobi_seq, twenty_three_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(23); }

TEST(kavtorev_d_iterative_jacobi_seq, twenty_four_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(24); }

TEST(kavtorev_d_iterative_jacobi_seq, twenty_five_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(25); }

TEST(kavtorev_d_iterative_jacobi_seq, twenty_seven_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(27); }

TEST(kavtorev_d_iterative_jacobi_seq, twenty_nine_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(29); }

TEST(kavtorev_d_iterative_jacobi_seq, thirty_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(30); }

TEST(kavtorev_d_iterative_jacobi_seq, thirty_one_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(31); }

TEST(kavtorev_d_iterative_jacobi_seq, fifty_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(50); }

TEST(kavtorev_d_iterative_jacobi_seq, hundred_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(100); }

TEST(kavtorev_d_iterative_jacobi_seq, thousand_random_matrix) { kavtorev_d_iterative_jacobi_seq::run_test(1000); }

TEST(kavtorev_d_iterative_jacobi_seq, null_task_data) {
  kavtorev_d_iterative_jacobi_seq::run_val(0, 0.0, 0, {}, {}, {});
}

TEST(kavtorev_d_iterative_jacobi_seq, insufficient_inputs) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, {}, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, null_input_pointers) {
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, {}, {}, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, null_output_pointers) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, {});
}

TEST(kavtorev_d_iterative_jacobi_seq, invalid_n) {
  std::vector<double> A(1, 1.0);
  std::vector<double> F(1, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(-1, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, invalid_eps) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, -0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, invalid_iterations) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, -10, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, null_matrix_data) {
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, {}, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, invalid_matrix_size) {
  std::vector<double> A(3, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, invalid_vector_size) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(3, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, output_size_less_than_n) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X(1, 0.0);
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, nan_in_matrix) {
  std::vector<double> A = {1.0, NAN, 0.0, 1.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, inf_in_matrix) {
  std::vector<double> A = {1.0, INFINITY, 0.0, 1.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, nan_in_vector) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F = {1.0, NAN};
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, inf_in_vector) {
  std::vector<double> A(4, 1.0);
  std::vector<double> F = {1.0, INFINITY};
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, zero_diagonal_element) {
  std::vector<double> A = {0.0, 1.0, 1.0, 0.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}

TEST(kavtorev_d_iterative_jacobi_seq, diagonal_not_dominant) {
  std::vector<double> A = {1.0, 2.0, 2.0, 1.0};
  std::vector<double> F(2, 1.0);
  std::vector<double> mpi_X;
  kavtorev_d_iterative_jacobi_seq::run_val(2, 0.01, 100, A, F, mpi_X);
}
