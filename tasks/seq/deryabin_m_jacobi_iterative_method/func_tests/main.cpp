#include <gtest/gtest.h>

#include "seq/deryabin_m_jacobi_iterative_method/include/ops_seq.hpp"

TEST(deryabin_m_jacobi_iterative_method_seq, test_simple_matrix) {
  // Create data
  std::vector<double> input_matrix_{6, 2, 3, 1, 5, 3, 1, 2, 4};
  std::vector<double> input_right_vector_{11, 9, 7};
  std::vector<double> output_x_vector_(3, 0);
  std::vector<double> true_solution{1, 1, 1};

  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  // Create Task
  deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential jacobi_iterative_method_TaskSequential(
      taskDataSeq);
  ASSERT_EQ(jacobi_iterative_method_TaskSequential.validation(), true);
  jacobi_iterative_method_TaskSequential.pre_processing();
  jacobi_iterative_method_TaskSequential.run();
  jacobi_iterative_method_TaskSequential.post_processing();
  ASSERT_EQ(true_solution, out_x_vec[0]);
}

TEST(deryabin_m_jacobi_iterative_method_seq, test_triangular_matrix) {
  // Create data
  std::vector<double> input_matrix_{16, 1, 2, 3,  4,  5,  0, 31, 6, 7, 8,  9,  0, 0, 34, 10, 11, 12,
                                    0,  0, 0, 28, 13, 14, 0, 0,  0, 0, 16, 15, 0, 0, 0,  0,  0,  17};
  std::vector<double> input_right_vector_{86, 202, 269, 261, 170, 102};
  std::vector<double> output_x_vector_(6, 0);
  std::vector<double> true_solution{1, 2, 3, 4, 5, 6};

  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  // Create Task
  deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential jacobi_iterative_method_TaskSequential(
      taskDataSeq);
  ASSERT_EQ(jacobi_iterative_method_TaskSequential.validation(), true);
  jacobi_iterative_method_TaskSequential.pre_processing();
  jacobi_iterative_method_TaskSequential.run();
  jacobi_iterative_method_TaskSequential.post_processing();
  ASSERT_EQ(true_solution, out_x_vec[0]);
}

TEST(deryabin_m_jacobi_iterative_method_seq, test_diagonal_elements_are_much_larger_than_non_diagonal) {
  // Create data
  std::vector<double> input_matrix_{999, 1,  2,  3,   4,  5,  6,  999, 7,  8,  9,   10, 11, 12, 999, 13, 14, 15,
                                    16,  17, 18, 999, 19, 20, 21, 22,  23, 24, 999, 25, 26, 27, 28,  29, 30, 999};
  std::vector<double> input_right_vector_{1069, 2162, 3244, 4315, 5375, 6424};
  std::vector<double> output_x_vector_(6, 0);
  std::vector<double> true_solution{1, 2, 3, 4, 5, 6};

  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  // Create Task
  deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential jacobi_iterative_method_TaskSequential(
      taskDataSeq);
  ASSERT_EQ(jacobi_iterative_method_TaskSequential.validation(), true);
  jacobi_iterative_method_TaskSequential.pre_processing();
  jacobi_iterative_method_TaskSequential.run();
  jacobi_iterative_method_TaskSequential.post_processing();
  ASSERT_EQ(true_solution, out_x_vec[0]);
}

TEST(deryabin_m_jacobi_iterative_method_seq, test_invalid_matrix_zeros_on_diagonal) {
  // Create data
  std::vector<double> input_matrix_{0,  1,  2,  3, 4,  5,  6,  0,  7,  8,  9, 10, 11, 12, 0,  13, 14, 15,
                                    16, 17, 18, 0, 19, 20, 21, 22, 23, 24, 0, 25, 26, 27, 28, 29, 30, 0};
  std::vector<double> input_right_vector_{70, 164, 247, 319, 380, 430};
  std::vector<double> output_x_vector_(6, 0);

  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  // Create Task
  deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential jacobi_iterative_method_TaskSequential(
      taskDataSeq);
  ASSERT_EQ(jacobi_iterative_method_TaskSequential.validation(), false);
}

TEST(deryabin_m_jacobi_iterative_method_seq, test_invalid_matrix_non_strict_diaganol_predominance) {
  // Create data
  std::vector<double> input_matrix_{15, 1,  2,  3,  4,  5,  6,  40, 7,  8,  9,   10, 11, 12, 65, 13, 14, 15,
                                    16, 17, 18, 90, 19, 20, 21, 22, 23, 24, 115, 25, 26, 27, 28, 29, 30, 140};
  std::vector<double> input_right_vector_{85, 244, 442, 679, 955, 1270};
  std::vector<double> output_x_vector_(6, 0);

  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  // Create Task
  deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential jacobi_iterative_method_TaskSequential(
      taskDataSeq);
  ASSERT_EQ(jacobi_iterative_method_TaskSequential.validation(), false);
}

TEST(deryabin_m_jacobi_iterative_method_seq, test_matrix_more_10_dimensions) {
  // Create data
  std::vector<double> input_matrix_(225, 0);
  std::vector<double> input_right_vector_(15, 1);
  for (unsigned short i = 0; i < 15; i++) {
    input_matrix_[i * 16] = 1;
  }
  std::vector<double> output_x_vector_(15, 0);
  std::vector<double> true_solution(15, 1);

  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  // Create Task
  deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential jacobi_iterative_method_TaskSequential(
      taskDataSeq);
  ASSERT_EQ(jacobi_iterative_method_TaskSequential.validation(), true);
  jacobi_iterative_method_TaskSequential.pre_processing();
  jacobi_iterative_method_TaskSequential.run();
  jacobi_iterative_method_TaskSequential.post_processing();
  ASSERT_EQ(true_solution, out_x_vec[0]);
}

TEST(deryabin_m_jacobi_iterative_method_seq, test_invalid_null_matrix) {
  // Create data
  std::vector<double> input_matrix_(36, 0);
  std::vector<double> input_right_vector_(6, 0);
  std::vector<double> output_x_vector_(6, 0);

  std::vector<std::vector<double>> in_matrix(1, input_matrix_);
  std::vector<std::vector<double>> in_right_part(1, input_right_vector_);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_right_part.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_x_vec.size());

  // Create Task
  deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential jacobi_iterative_method_TaskSequential(
      taskDataSeq);
  ASSERT_EQ(jacobi_iterative_method_TaskSequential.validation(), false);
}
