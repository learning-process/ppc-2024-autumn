#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "seq/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_seq.hpp"

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_2_unknowns) {
  const int count = 2;

  // Create data
  std::vector<double> in = {1, -1, -5, 2, 1, -7};
  std::vector<double> out(count, 0);
  std::vector<double> ans = {-4, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
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

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_3_unknowns) {
  const int count = 3;

  // Create data
  std::vector<double> in = {3, 2, -5, -1, 2, -1, 3, 13, 1, 2, -1, 9};
  std::vector<double> out(count, 0);
  std::vector<double> ans = {3, 5, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
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

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_4_unknowns) {
  const int count = 4;

  // Create data
  std::vector<double> in = {2, 5, 4, 1, 20, 1, 3, 2, 1, 11, 2, 10, 9, 7, 40, 3, 8, 9, 2, 37};
  std::vector<double> out(count, 0);
  std::vector<double> ans = {1, 2, 2, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
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

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_10_unknowns) {
  const int count = 10;

  // Create data
  std::vector<double> in(count * (count + 1));
  std::vector<double> out(count, 0);
  std::vector<double> ans(count);

  for (int i = 0; i < count; ++i) {
    in[i * (count + 1) + i] = 1;
    in[i * (count + 1) + count] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
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

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_50_unknowns) {
  const int count = 50;

  // Create data
  std::vector<double> in(count * (count + 1));
  std::vector<double> out(count, 0);
  std::vector<double> ans(count);

  for (int i = 0; i < count; ++i) {
    in[i * (count + 1) + i] = 1;
    in[i * (count + 1) + count] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
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

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_100_unknowns) {
  const int count = 100;

  // Create data
  std::vector<double> in(count * (count + 1));
  std::vector<double> out(count, 0);
  std::vector<double> ans(count);

  for (int i = 0; i < count; ++i) {
    in[i * (count + 1) + i] = 1;
    in[i * (count + 1) + count] = i + 1;
  }
  std::iota(ans.begin(), ans.end(), 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(count);
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