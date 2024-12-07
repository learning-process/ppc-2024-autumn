#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "seq/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_seq.hpp"

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_for_empty_matrix) {
  const int cols = 0;
  const int rows = 0;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_when_matrix_is_not_square) {
  const int cols = 5;
  const int rows = 2;

  // Create data
  std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> out(cols - 1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_when_determinant_is_0) {
  const int cols = 4;
  const int rows = 3;

  // Create data
  std::vector<double> in = {6, -1, 12, 3, -3, -5, -6, 9, 1, 4, 2, -1};
  std::vector<double> out(cols - 1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_when_ranks_are_not_equal) {
  const int cols = 4;
  const int rows = 3;

  // Create data
  std::vector<double> in = {1, 2, 3, 7, 4, 5, 6, 2, 7, 8, 9, 8};
  std::vector<double> out(cols - 1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_3x2) {
  const int cols = 3;
  const int rows = 2;

  // Create data
  std::vector<double> in = {1, -1, -5, 2, 1, -7};
  std::vector<double> out(cols - 1, 0);
  std::vector<double> ans = {-4, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_4x3) {
  const int cols = 4;
  const int rows = 3;

  // Create data
  std::vector<double> in = {3, 2, -5, -1, 2, -1, 3, 13, 1, 2, -1, 9};
  std::vector<double> out(cols - 1, 0);
  std::vector<double> ans = {3, 5, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_5x4) {
  const int cols = 5;
  const int rows = 4;

  // Create data
  std::vector<double> in = {1, 1, 2, 3, 1, 1, 2, 3, -1, -4, 3, -1, -1, -2, -4, 2, 3, -1, -1, -6};
  std::vector<double> out(cols - 1, 0);
  std::vector<double> ans = {-1, -1, 0, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_11x10) {
  const int cols = 11;
  const int rows = 10;

  // Create data
  std::vector<double> in(cols * rows);
  std::vector<double> out(cols - 1, 0);
  std::vector<double> ans(cols - 1);

  for (int i = 0; i < rows; ++i) {
    in[i * cols + i] = 1;
    in[i * cols + rows] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_51x50) {
  const int cols = 51;
  const int rows = 50;

  // Create data
  std::vector<double> in(cols * rows);
  std::vector<double> out(cols - 1, 0);
  std::vector<double> ans(cols - 1);

  for (int i = 0; i < rows; ++i) {
    in[i * cols + i] = 1;
    in[i * cols + rows] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_101x100) {
  const int cols = 101;
  const int rows = 100;

  // Create data
  std::vector<double> in(cols * rows);
  std::vector<double> out(cols - 1, 0);
  std::vector<double> ans(cols - 1);

  for (int i = 0; i < rows; ++i) {
    in[i * cols + i] = 1;
    in[i * cols + rows] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}