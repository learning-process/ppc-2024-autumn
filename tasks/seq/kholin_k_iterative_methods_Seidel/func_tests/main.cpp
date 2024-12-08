#include <gtest/gtest.h>

#include <seq/kholin_k_iterative_methods_Seidel/src/ops_seq.cpp>

#include "seq/kholin_k_iterative_methods_Seidel/include/ops_seq.hpp"

TEST(kholin_k_iterative_methods_Seidel_seq, validation_true_when_matrix_with_diag_pred) {
  const size_t count_rows = 3;
  const size_t count_colls = 3;
  float epsilon = 0.001f;

  std::vector<float> in = kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);
  std::vector<float> out(count_rows, 0.0f);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  kholin_k_iterative_methods_Seidel_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
}
TEST(kholin_k_iterative_methods_Seidel_seq, test_pre_processing) {
  const size_t count_rows = 3;
  const size_t count_colls = 3;
  float epsilon = 0.001f;

  std::vector<float> in = kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);
  std::vector<float> out(count_rows, 0.0f);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  kholin_k_iterative_methods_Seidel_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
}
TEST(kholin_k_iterative_methods_Seidel_seq, test_run) {
  const size_t count_rows = 3;
  const size_t count_colls = 3;
  float epsilon = 0.001f;

  std::vector<float> in = kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);
  std::vector<float> out(count_rows, 0.0f);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  kholin_k_iterative_methods_Seidel_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
}

TEST(kholin_k_iterative_methods_Seidel_seq, test_post_processing) {
  const size_t count_rows = 3;
  const size_t count_colls = 3;
  float epsilon = 0.001f;

  std::vector<float> in = kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);
  std::vector<float> out(count_rows, 0.0f);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  kholin_k_iterative_methods_Seidel_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  ASSERT_EQ(testTaskSequential.post_processing(), true);
}

TEST(kholin_k_iterative_methods_Seidel_seq, validation_false_when_matrix_no_quadro) {
  const size_t count_rows = 3;
  const size_t count_colls = 4;
  float epsilon = 0.001f;

  std::vector<float> in = kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(count_rows, count_colls);
  std::vector<float> out(count_rows, 0.0f);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  kholin_k_iterative_methods_Seidel_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kholin_k_iterative_methods_Seidel_seq, validation_false_when_matrix_without_diag_pred_ver1) {
  const size_t count_rows = 3;
  const size_t count_colls = 3;
  float epsilon = 0.001f;

  std::vector<float> in(count_rows * count_colls);
  bool IsValid = false;
  do {
    int count = 0;
    for (size_t i = 0; i < count_rows; i++) {
      for (size_t j = 0; j < count_colls; j++) {
        in[count_colls * i + j] = kholin_k_iterative_methods_Seidel_seq::gen_float_value();
      }
      if (kholin_k_iterative_methods_Seidel_seq::IsDiagPred(in, count_colls, count_colls * i, count_colls * i + i)) {
        count++;
      }
    }
    if (count == count_rows) {
      IsValid = true;
    }
  } while (IsValid);
  std::vector<float> out(count_rows);
  std::vector<float> X0(count_rows, 0.0f);
  std::vector<float> B = kholin_k_iterative_methods_Seidel_seq::gen_vector(count_rows);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_colls);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(count_rows);

  kholin_k_iterative_methods_Seidel_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), IsValid);
}