#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/belov_a_gauss_seidel_iter_method/include/ops_seq.hpp"

using namespace belov_a_gauss_seidel_seq;

namespace belov_a_gauss_seidel_seq {
std::vector<double> generateDiagonallyDominantMatrix(int n) {
  std::vector<double> A_local(n * n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A_local[i * n + j] = dis(gen);
        row_sum += abs(A_local[i * n + j]);
      }
    }
    A_local[i * n + i] = row_sum + abs(dis(gen)) + 1.0;
  }
  return A_local;
}

std::vector<double> generateFreeMembers(int n) {
  std::vector<double> freeMembers(n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    freeMembers[i] = dis(gen);
  }
  return freeMembers;
}
}  // namespace belov_a_gauss_seidel_seq

TEST(belov_a_gauss_seidel_seq, test_int_sample1_SLAE) {
  int n = 3;
  double epsilon = 0.05;

  std::vector<double> input_matrix = {10, -3, 2, 3, -10, -2, 2, -3, 10};
  std::vector<double> freeMembersVector = {10, -23, 26};
  std::vector<double> solutionVector(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());

  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<double> result = {1, 2, 3};

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(result[i], solutionVector[i], epsilon);
  }
}

TEST(belov_a_gauss_seidel_seq, test_non_square_matrix) {
  int n = 3;
  double epsilon = 0.05;

  std::vector<double> input_matrix = {10, -3, 2, 3, -10, -2};
  std::vector<double> freeMembersVector = {10, -23, 26};
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_gauss_seidel_seq, test_no_diagonal_dominance) {
  int n = 3;
  double epsilon = 0.1;

  std::vector<double> input_matrix = {1, 3, 1, 1, 1, 1, 1, 3, 1};
  std::vector<double> freeMembersVector = {5, 5, 5};
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_gauss_seidel_seq, test_large_SLAE) {
  int n = 80;
  double epsilon = 0.1;

  std::vector<double> input_matrix(n * n, 0);
  for (int i = 0; i < n; ++i) {
    input_matrix[i * n + i] = n;
  }
  std::vector<double> freeMembersVector(n, 1);
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());

  testTaskSequential.run();
  testTaskSequential.post_processing();
}

TEST(belov_a_gauss_seidel_seq, test_double_sample2_SLAE) {
  int n = 3;
  double epsilon = 0.00025;

  std::vector<double> input_matrix = {6, -1, -1, -1, 6, -1, -1, -1, 6};
  std::vector<double> freeMembersVector = {11.33, 32.00, 42.00};
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());

  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<double> expected_result = {4.66607143, 7.61892857, 9.0475};

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(expected_result[i], solutionVector[i], epsilon);
  }
}

TEST(belov_a_gauss_seidel_seq, test_double_sample3_SLAE_4x4) {
  int n = 4;
  double epsilon = 0.0001;

  std::vector<double> input_matrix = {3.82, 1.02, 0.75, 0.81, 1.05, 4.53, 0.98, 1.53,
                                      0.73, 0.85, 4.71, 0.81, 0.88, 0.81, 1.28, 3.50};
  std::vector<double> freeMembersVector = {15.655, 22.705, 23.480, 16.110};
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());

  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<double> expected_result = {2.12727865, 3.03258813, 3.76611894, 1.98884747};

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(expected_result[i], solutionVector[i], epsilon);
  }
}

TEST(belov_a_gauss_seidel_seq, test_validation_empty_data) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_gauss_seidel_seq, test_invalid_input_matrix_size) {
  int n = 3;
  double epsilon = 0.05;

  std::vector<double> input_matrix = {10, -3, 2};
  std::vector<double> freeMembersVector = {10, -23, 26};
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_gauss_seidel_seq, test_invalid_inputs_count) {
  int n = 8;
  double epsilon = 0.004;

  std::vector<double> input_matrix = generateDiagonallyDominantMatrix(n);
  std::vector<double> freeMembersVector = generateFreeMembers(n);
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  /*taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size()); // "forgot" to fill inputs_count
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());*/
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_gauss_seidel_seq, test_small_system_non_diagonal_dominant) {
  const int n = 3;
  double epsilon = 1e-6;

  std::vector<double> input_matrix = {1, 1, 1, 1, 1, 1, 1, 1, 1};  // no diagonal dominance
  std::vector<double> freeMembersVector = {3, 3, 3};
  std::vector<double> solutionVector(n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_gauss_seidel_seq, test_ge_double_equal_numbers) {
  double a = 1.0;
  double b = 1.0;
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_seq, test_ge_double_small_difference) {
  double a = 1.0;
  double b = 1.0 + std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_seq, test_ge_double_large_difference) {
  double a = 10.0;
  double b = 5.0;
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_seq, test_ge_double_negative_numbers) {
  double a = -1.0;
  double b = -1.0 - std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_seq, test_ge_double_zero_numbers) {
  double a = 0.0;
  double b = 0.0;
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}