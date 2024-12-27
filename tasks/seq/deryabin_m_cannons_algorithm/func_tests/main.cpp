#include <gtest/gtest.h>

#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

TEST(deryabin_m_cannons_algorithm_seq, test_simple_matrix) {
  // Create data
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_B{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> output_matrix_C(9, 0);
  std::vector<double> true_solution{30, 36, 42, 66, 81, 96, 102, 126, 150};

  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_TaskSequential(taskDataSeq);
  ASSERT_EQ(cannons_algorithm_TaskSequential.validation(), true);
  cannons_algorithm_TaskSequential.pre_processing();
  cannons_algorithm_TaskSequential.run();
  cannons_algorithm_TaskSequential.post_processing();
  ASSERT_EQ(true_solution, out_matrix_C[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_triangular_matrix) {
  // Create data
  std::vector<double> input_matrix_A{1, 2, 3, 0, 5, 6, 0, 0, 9};
  std::vector<double> input_matrix_B{1, 0, 0, 4, 5, 0, 7, 8, 9};
  std::vector<double> output_matrix_C(9, 0);
  std::vector<double> true_solution{30, 34, 27, 62, 73, 54, 63, 72, 81};

  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_TaskSequential(taskDataSeq);
  ASSERT_EQ(cannons_algorithm_TaskSequential.validation(), true);
  cannons_algorithm_TaskSequential.pre_processing();
  cannons_algorithm_TaskSequential.run();
  cannons_algorithm_TaskSequential.post_processing();
  ASSERT_EQ(true_solution, out_matrix_C[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_null_matrix) {
  // Create data
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_B(9, 0);
  std::vector<double> output_matrix_C(9, 0);

  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_TaskSequential(taskDataSeq);
  ASSERT_EQ(cannons_algorithm_TaskSequential.validation(), true);
  cannons_algorithm_TaskSequential.pre_processing();
  cannons_algorithm_TaskSequential.run();
  cannons_algorithm_TaskSequential.post_processing();
  ASSERT_EQ(in_matrix_B[0], out_matrix_C[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_identity_matrix) {
  // Create data
  std::vector<double> input_matrix_A{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> input_matrix_B{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> output_matrix_C(9, 0);

  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_TaskSequential(taskDataSeq);
  ASSERT_EQ(cannons_algorithm_TaskSequential.validation(), true);
  cannons_algorithm_TaskSequential.pre_processing();
  cannons_algorithm_TaskSequential.run();
  cannons_algorithm_TaskSequential.post_processing();
  ASSERT_EQ(in_matrix_B[0], out_matrix_C[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_matrices_of_different_dimensions) {
  // Create data
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_B{1, 2, 3, 4};
  std::vector<double> output_matrix_C(9, 0);

  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_TaskSequential(taskDataSeq);
  ASSERT_EQ(cannons_algorithm_TaskSequential.validation(), false);
}

TEST(deryabin_m_cannons_algorithm_seq, test_non_square_matrices) {
  // Create data
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5};
  std::vector<double> input_matrix_B{1, 2, 3, 4, 5};
  std::vector<double> output_matrix_C(5, 0);

  std::vector<std::vector<double>> in_matrix_A(1, input_matrix_A);
  std::vector<std::vector<double>> in_matrix_B(1, input_matrix_B);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
  taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

  // Create Task
  deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential cannons_algorithm_TaskSequential(taskDataSeq);
  ASSERT_EQ(cannons_algorithm_TaskSequential.validation(), false);
}
