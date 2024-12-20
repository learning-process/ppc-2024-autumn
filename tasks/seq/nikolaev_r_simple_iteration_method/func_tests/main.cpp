#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include "seq/nikolaev_r_simple_iteration_method/include/ops_seq.hpp"

TEST(nikolaev_r_simple_iteration_method_seq, test_2x2_matrix) {
  const size_t m_size = 2;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {5.0, 3.0, 4.0, 6.0};
  std::vector<double> vect = {2.0, 6.0};
  std::vector<double> expected_solution = {-0.333, 1.222};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-3);
  }
}

TEST(nikolaev_r_simple_iteration_method_seq, test_3x3_matrix) {
  const size_t m_size = 3;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {4.0, 2.0, 1.0, 3.0, 7.0, -2.0, -2.0, 3.0, 8.0};
  std::vector<double> vect = {-2.0, 5.0, 1.0};
  std::vector<double> expected_solution = {-0.866, 0.957, -0.450};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-3);
  }
}

TEST(nikolaev_r_simple_iteration_method_seq, test_4x4_matrix) {
  const size_t m_size = 4;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {5.0, -2.0, 1.0, 1.0, -2.0, 7.0, 3.0, 1.0, -1.0, 1.0, 9.0, 6.0, 1.0, 5.0, -3.0, 14.0};
  std::vector<double> vect = {4.0, -5.0, 1.0, 2.0};
  std::vector<double> expected_solution = {0.479, -0.629, 0.010, 0.336};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-3);
  }
}

TEST(nikolaev_r_simple_iteration_method_seq, test_5x5_matrix) {
  const size_t m_size = 5;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {9.0, -2.0, 1.0, 1.0, 3.0,  -2.0, 16.0, 3.0, 1.0,  -5.0, -1.0, 1.0, 19.0,
                              6.0, 1.0,  1.0, 5.0, -3.0, 14.0, 2.0,  6.0, -1.0, -5.0, 1.0,  15.0};
  std::vector<double> vect = {-5.0, 6.0, 2.0, -4.0, 5.0};
  std::vector<double> expected_solution = {-0.641, 0.517, 0.162, -0.491, 0.711};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-3);
  }
}

TEST(nikolaev_r_simple_iteration_method_seq, test_identity_matrix) {
  const size_t m_size = 3;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> vect = {1.0, -1.0, 1.0};
  std::vector<double> expected_solution = {1.0, -1.0, 1.0};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out[i], expected_solution[i], 1e-3);
  }
}

TEST(nikolaev_r_simple_iteration_method_seq, test_non_diagonally_dominant_matrix) {
  const size_t m_size = 3;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {4.0, 5.0, 1.0, 3.0, 7.0, -2.0, -2.0, 3.0, 8.0};
  std::vector<double> vect = {-2.0, 5.0, 1.0};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(nikolaev_r_simple_iteration_method_seq, test_singular_matrix) {
  const size_t m_size = 3;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 3.0, 6.0, 9.0};
  std::vector<double> vect = {-2.0, 5.0, 1.0};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(nikolaev_r_simple_iteration_method_seq, test_incorrect_matrix_size) {
  const size_t m_size = 0;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {4.0, 2.0, 1.0, 3.0, 7.0, -2.0, -2.0, 3.0, 8.0};
  std::vector<double> vect = {-2.0, 5.0, 1.0};
  std::vector<double> expected_solution = {-0.866, 0.957, -0.450};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(nikolaev_r_simple_iteration_method_seq, test_incorrect_input_data) {
  const size_t m_size = 3;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {4.0, 2.0, 1.0, 3.0, 7.0, -2.0, -2.0, 3.0, 8.0};
  std::vector<double> vect = {-2.0, 5.0, 1.0};
  std::vector<double> expected_solution = {-0.866, 0.957, -0.450};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(nikolaev_r_simple_iteration_method_seq, test_incorrect_output_data) {
  const size_t m_size = 3;
  std::vector<size_t> in(1, m_size);
  std::vector<double> matr = {4.0, 2.0, 1.0, 3.0, 7.0, -2.0, -2.0, 3.0, 8.0};
  std::vector<double> vect = {-2.0, 5.0, 1.0};
  std::vector<double> expected_solution = {-0.866, 0.957, -0.450};
  std::vector<double> out(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
  taskDataSeq->inputs_count.emplace_back(matr.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  taskDataSeq->inputs_count.emplace_back(vect.size());

  nikolaev_r_simple_iteration_method_seq::SimpleIterationMethodSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}