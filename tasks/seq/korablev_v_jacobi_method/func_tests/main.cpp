#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "seq/korablev_v_jacobi_method/include/ops_seq.hpp"

TEST(korablev_v_jacobi_method, test_matrix_2x2) {
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {4.0, 1.0, 2.0, 3.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> expected_solution = {0.1, 0.6};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_TRUE(jacobiTaskSequential.validation());
  jacobiTaskSequential.pre_processing();
  jacobiTaskSequential.run();
  jacobiTaskSequential.post_processing();

  for (size_t i = 0; i < matrix_size; ++i) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-6);
  }
}

TEST(korablev_v_jacobi_method, test_matrix_3x3) {
  const size_t matrix_size = 3;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {10.0, 1.0, 2.0, 1.0, 8.0, 1.0, 2.0, 1.0, 9.0};
  std::vector<double> vector_data = {7.0, -4.0, 5.0};
  std::vector<double> expected_solution = {0.66864785, -0.64338782, 0.47845468};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_TRUE(jacobiTaskSequential.validation());
  jacobiTaskSequential.pre_processing();
  jacobiTaskSequential.run();
  jacobiTaskSequential.post_processing();

  for (size_t i = 0; i < matrix_size; ++i) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-6);
  }
}

TEST(korablev_v_jacobi_method, test_matrix_4x4) {
  const size_t matrix_size = 4;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {10.0, -1.0, 2.0,  0.0,  -1.0, 11.0, -1.0, 3.0,
                                     2.0,  -1.0, 10.0, -1.0, 0.0,  3.0,  -1.0, 8.0};
  std::vector<double> vector_data = {6.0, 25.0, -11.0, 15.0};
  std::vector<double> expected_solution = {1.0, 2.0, -1.0, 1.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_TRUE(jacobiTaskSequential.validation());
  jacobiTaskSequential.pre_processing();
  jacobiTaskSequential.run();
  jacobiTaskSequential.post_processing();

  for (size_t i = 0; i < matrix_size; ++i) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-6);
  }
}

TEST(korablev_v_jacobi_method, test_matrix_5x5) {
  const size_t matrix_size = 5;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {15.0, -1.0, 2.0, 0.0, 3.0,  -1.0, 14.0, -1.0, 3.0, 0.0, 2.0,  -1.0, 15.0,
                                     -1.0, 2.0,  0.0, 3.0, -1.0, 13.0, -1.0, 3.0,  0.0, 2.0, -1.0, 14.0};
  std::vector<double> vector_data = {7.0, 19.0, 5.0, -10.0, 14.0};
  std::vector<double> expected_solution = {0.38480008, 1.62822036, 0.21132843, -1.06631886, 0.81118743};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_TRUE(jacobiTaskSequential.validation());
  jacobiTaskSequential.pre_processing();
  jacobiTaskSequential.run();
  jacobiTaskSequential.post_processing();

  for (size_t i = 0; i < matrix_size; ++i) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-5);
  }
}

TEST(korablev_v_jacobi_method, test_matrix_6x6) {
  const size_t matrix_size = 6;
  std::vector<size_t> in_size(1, matrix_size);

  std::vector<double> matrix_data = {15.0, -1.0, 2.0,  0.0,  3.0,  -2.0, -1.0, 18.0, -1.0, 3.0,  0.0,  2.0,
                                     2.0,  -1.0, 15.0, -1.0, 2.0,  -1.0, 0.0,  3.0,  -1.0, 13.0, -1.0, 1.0,
                                     3.0,  0.0,  2.0,  -1.0, 14.0, -2.0, -2.0, 2.0,  -1.0, 1.0,  -2.0, 17.0};

  std::vector<double> vector_data = {5.0, 12.0, 7.0, -3.0, 10.0, -8.0};
  std::vector<double> expected_solution = {0.16989913316691407, 0.7976844699507312, 0.3752216234950149,
                                           -0.3106171008637048, 0.5391352751989575, -0.44067419993005874};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_TRUE(jacobiTaskSequential.validation());
  jacobiTaskSequential.pre_processing();
  jacobiTaskSequential.run();
  jacobiTaskSequential.post_processing();

  for (size_t i = 0; i < matrix_size; ++i) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-6);
  }
}

TEST(korablev_v_jacobi_method, invalid_input_count) {
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {4.0, 1.0, 2.0, 3.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_FALSE(jacobiTaskSequential.validation());
}

TEST(korablev_v_jacobi_method, invalid_output_count) {
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {4.0, 1.0, 2.0, 3.0};
  std::vector<double> vector_data = {1.0, 2.0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_FALSE(jacobiTaskSequential.validation());
}

TEST(korablev_v_jacobi_method, non_diagonally_dominant_matrix) {
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {1.0, 2.0, 2.0, 1.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_FALSE(jacobiTaskSequential.validation());
}

TEST(korablev_v_jacobi_method, zero_on_diagonal) {
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {0.0, 1.0, 1.0, 3.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);
  ASSERT_FALSE(jacobiTaskSequential.validation());
}

TEST(korablev_v_jacobi_method, invalid_matrix_size) {
  const size_t matrix_size = 3;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {4.0, 1.0, 2.0, 3.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> out(2, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(sizeof(size_t));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_data.size() * sizeof(double));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
  taskDataSeq->inputs_count.emplace_back(vector_data.size() * sizeof(double));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size() * sizeof(double));

  korablev_v_jacobi_method_seq::JacobiMethodSequential jacobiTaskSequential(taskDataSeq);

  ASSERT_FALSE(jacobiTaskSequential.validation());
}