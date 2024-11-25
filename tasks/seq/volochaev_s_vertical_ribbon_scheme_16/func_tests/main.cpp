#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/volochaev_s_vertical_ribbon_scheme_16/include/ops_seq.hpp"

namespace volochaev_s_vertical_ribbon_scheme_16_seq {

void get_random_matrix(std::vector<int> &mat) {
  std::random_device dev;
  std::mt19937 gen(dev());

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = 50 - gen() % 100;
  }
}
}  // namespace volochaev_s_vertical_ribbon_scheme_16_seq

TEST(volochaev_s_vertical_ribbon_scheme_16_seq, Test_0) {
  // Create data
  std::vector<int> in;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(volochaev_s_vertical_ribbon_scheme_16_seq, Test_1) {
  // Create data
  std::vector<int> in_A(10, 0);
  std::vector<int> in_B(2, 0);
  std::vector<int> out(5, 0);
  std::vector<int> ans(5, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(volochaev_s_vertical_ribbon_scheme_16_seq, Test_2) {
  // Create data
  std::vector<int> in_A(10, 0);
  std::vector<int> in_B(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());


  // Create Task
  volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(volochaev_s_vertical_ribbon_scheme_16_seq, Test_3) {
  // Create data

  std::vector<int> in_A(10, 0);
  volochaev_s_vertical_ribbon_scheme_16_seq::get_random_matrix(in_A);
  std::vector<int> in_B(2, 0);
  std::vector<int> out(5, 0);
  std::vector<int> ans(5, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());


  // Create Task
  volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(volochaev_s_vertical_ribbon_scheme_16_seq, Test_4) {
  // Create data
  std::vector<int> in_A(10, 0);
  std::vector<int> in_B(2, 0);
  volochaev_s_vertical_ribbon_scheme_16_seq::get_random_matrix(in_B);
  std::vector<int> out(5, 0);
  std::vector<int> ans(5, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(volochaev_s_vertical_ribbon_scheme_16_seq, Test_5) {
  // Create data
  std::vector<int> in_A(10, 0);
  std::vector<int> in_B;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  // Create Task
  volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(volochaev_s_vertical_ribbon_scheme_16_seq, Test_6) {
  // Create data
  std::vector<int> in_A(12, 1);
  std::vector<int> in_B(4, 1);
  std::vector<int> out(3, 0);
  std::vector<int> ans(3, 4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_vertical_ribbon_scheme_16_seq::Lab2_16 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}
