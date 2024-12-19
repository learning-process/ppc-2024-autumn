#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/volochaev_s_shell_sort_with_simple_merge_16/include/ops_seq.hpp"

namespace volochaev_s_shell_sort_with_simple_merge_16_seq {

void get_random_matrix(std::vector<int> &mat, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = dis(gen);
  }
}

}  // namespace volochaev_s_shell_sort_with_simple_merge_16_seq

TEST(volochaev_s_shell_sort_with_simple_merge_16_seq, Test_mines_1) {
  std::vector<int> global_A(100);

  ASSERT_ANY_THROW(volochaev_s_shell_sort_with_simple_merge_16_seq::get_random_matrix(global_A, 90, -100));
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_seq, Test_0) {
  // Create data
  std::vector<int> in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  // Create Task
  volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_seq, Test_1) {
  // Create data

  std::vector<int> in_A(10, 0);
  volochaev_s_shell_sort_with_simple_merge_16_seq::get_random_matrix(in_A, -100, 100);
  std::vector<int> out(10, 0);
  std::vector<int> ans = in_A;
  std::sort(ans.begin(), ans.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_seq, Test_3) {
  // Create data

  std::vector<int> in_A(1, 0);
  volochaev_s_shell_sort_with_simple_merge_16_seq::get_random_matrix(in_A, -100, 100);
  std::vector<int> out(1, 0);
  std::vector<int> ans = in_A;
  std::sort(ans.begin(), ans.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_seq, Test_4) {
  // Create data

  std::vector<int> in_A(13, 0);
  volochaev_s_shell_sort_with_simple_merge_16_seq::get_random_matrix(in_A, -100, 100);
  std::vector<int> out(13, 0);
  std::vector<int> ans = in_A;
  std::sort(ans.begin(), ans.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_seq, Test_5) {
  // Create data

  std::vector<int> in_A(20, 0);
  volochaev_s_shell_sort_with_simple_merge_16_seq::get_random_matrix(in_A, -100, 100);
  std::vector<int> out(20, 0);
  std::vector<int> ans = in_A;
  std::sort(ans.begin(), ans.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_seq, Test_6) {
  // Create data

  std::vector<int> in_A(100, 0);
  volochaev_s_shell_sort_with_simple_merge_16_seq::get_random_matrix(in_A, -100, 100);
  std::vector<int> out(100, 0);
  std::vector<int> ans = in_A;
  std::sort(ans.begin(), ans.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}
