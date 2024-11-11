#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/belov_a_gauss_seidel_iter_method/include/ops_seq.hpp"

using namespace belov_a_gauss_seidel_seq;

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
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  EXPECT_TRUE(testTaskSequential.validation());
  EXPECT_TRUE(testTaskSequential.pre_processing());

  ASSERT_FALSE(testTaskSequential.run());
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
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionVector.data()));

  GaussSeidelSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

// TEST(belov_a_gauss_seidel_seq, Test_SmallSystem_NegativeIntegers) {
//   const int n = 3;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix = {4, -1, -1, -1, 3, -1, -1, -1, 5};
//   std::vector<double> b = {-7, -8, -10};
//   std::vector<double> out(n, 0.0);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_TRUE(testTaskSequential.validation());
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//
//   ASSERT_NEAR(out[0], -1.0, 0.01);
//   ASSERT_NEAR(out[1], -2.0, 0.01);
//   ASSERT_NEAR(out[2], -1.0, 0.01);
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_MediumSystem_MixedValues) {
//   const int n = 4;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};
//   std::vector<double> b = {6, 25, -11, 15};
//   std::vector<double> out(n, 0.0);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_TRUE(testTaskSequential.validation());
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//
//   ASSERT_NEAR(out[0], 1.0, 0.01);
//   ASSERT_NEAR(out[1], 2.0, 0.01);
//   ASSERT_NEAR(out[2], -1.0, 0.01);
//   ASSERT_NEAR(out[3], 1.0, 0.01);
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_Validation_Fails_NoDiagonalDominance) {
//   const int n = 3;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix = {1, 1, 1, 1, 1, 1, 1, 1, 1};  // matrix without diagonal dominance
//   std::vector<double> b = {3, 3, 3};
//   std::vector<double> out(n, 0.0);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_FALSE(testTaskSequential.validation());
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_SingleEquation) {
//   const int n = 1;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix = {5};
//   std::vector<double> b = {10};
//   std::vector<double> out(n, 0.0);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_TRUE(testTaskSequential.validation());
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//
//   ASSERT_NEAR(out[0], 2.0, 0.01);
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_LargeSystem_RandomValues) {
//   const int n = 50;
//   double epsilon = 1e-5;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix(n * n, 0.0);
//   std::vector<double> b(n, 1.0);
//   std::vector<double> out(n, 0.0);
//
//   for (int i = 0; i < n; ++i) {
//     for (int j = 0; j < n; ++j) {
//       if (i == j) {
//         matrix[i * n + j] = 10.0;
//       } else {
//         matrix[i * n + j] = (i % 2 == 0) ? -1.0 : 1.0;
//       }
//     }
//   }
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_TRUE(testTaskSequential.validation());
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//
//   for (const auto& val : out) {
//     ASSERT_NEAR(val, 0.1, 0.05);
//   }
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_Validation_Fails_InvalidMatrixSize) {
//   const int n = 2;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix = {4, 1};  // wrong size of the matrix
//   std::vector<double> b = {5, 6};
//   std::vector<double> out(n, 0.0);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_FALSE(testTaskSequential.validation());
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_SmallSystem_NonDiagonalDominant) {
//   const int n = 3;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix = {1, 1, 1, 1, 1, 1, 1, 1, 1};  // matrix without diagonal dominance
//   std::vector<double> b = {3, 3, 3};
//   std::vector<double> out(n, 0.0);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_FALSE(testTaskSequential.validation());
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_VeryLargeSystem_DiagonalMatrix) {
//   const int n = 100;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix(n * n, 0.0);
//   std::vector<double> b(n, 1.0);
//   std::vector<double> out(n, 0.0);
//
//   for (int i = 0; i < n; ++i) {
//     matrix[i * n + i] = 10.0;
//   }
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_TRUE(testTaskSequential.validation());
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//
//   for (const auto& val : out) {
//     ASSERT_NEAR(val, 0.1, 0.01);
//   }
// }
//
// TEST(belov_a_gauss_seidel_seq, Test_SingleEquation_SolutionCheck) {
//   const int n = 1;
//   double epsilon = 1e-6;
//
//   std::vector<int> dimensions = {n};
//   std::vector<double> matrix = {7.0};  // matrix 1x1
//   std::vector<double> b = {14.0};
//   std::vector<double> out(n, 0.0);
//
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   GaussSeidelSequential testTaskSequential(taskDataSeq);
//   ASSERT_TRUE(testTaskSequential.validation());
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//
//   ASSERT_NEAR(out[0], 2.0, 0.01);
// }
