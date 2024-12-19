// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/muhina_m_shell_sort/include/ops_seq.hpp"

namespace muhina_m_shell_sort_mpi {

std::vector<int> Get_Random_Vector(int sz, int min_value, int max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min_value + gen() % (max_value - min_value + 1);
  }
  return vec;
}
}  // namespace muhina_m_shell_sort_mpi

TEST(muhina_m_shell_sort_seq, Test_Sort_Small) {
  std::vector<int> in = {5, 2, 9, 1, 5, 6};
  std::vector<int> out(in.size(), 0);
  std::vector<int> expected_result = {1, 2, 5, 5, 6, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muhina_m_shell_sort_seq::ShellSortSequential ShellSortSequential(taskDataSeq);
  ASSERT_EQ(ShellSortSequential.validation(), true);
  ShellSortSequential.pre_processing();
  ShellSortSequential.run();
  ShellSortSequential.post_processing();
  ASSERT_EQ(out, expected_result);
}

TEST(muhina_m_shell_sort_seq, Test_EmptyVec) {
  std::vector<int> in = {};
  std::vector<int> out(in.size());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muhina_m_shell_sort_seq::ShellSortSequential ShellSortSequential(taskDataSeq);
  EXPECT_FALSE(ShellSortSequential.validation());
}

TEST(muhina_m_shell_sort_seq, Test_Sort_RepeatingValues) {
  std::vector<int> in = {1, 1, 1, 1, 1, 1};
  std::vector<int> out(in.size(), 0);
  std::vector<int> expected_result = {1, 1, 1, 1, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muhina_m_shell_sort_seq::ShellSortSequential ShellSortSequential(taskDataSeq);
  ASSERT_EQ(ShellSortSequential.validation(), true);
  ShellSortSequential.pre_processing();
  ShellSortSequential.run();
  ShellSortSequential.post_processing();
  ASSERT_EQ(out, expected_result);
}

TEST(muhina_m_shell_sort_seq, Test_Sort_NegativeValues) {
  std::vector<int> in = {-5, -2, -9, -1, -5, -6};
  std::vector<int> out(in.size(), 0);
  std::vector<int> expected_result = {-9, -6, -5, -5, -2, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muhina_m_shell_sort_seq::ShellSortSequential ShellSortSequential(taskDataSeq);
  ASSERT_EQ(ShellSortSequential.validation(), true);
  ShellSortSequential.pre_processing();
  ShellSortSequential.run();
  ShellSortSequential.post_processing();
  ASSERT_EQ(out, expected_result);
}

TEST(muhina_m_shell_sort_seq, Test_Sort_RandomValues) {
  const int count = 10;
  const int min_val = 0;
  const int max_val = 100;
  std::vector<int> in = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected_result = in;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muhina_m_shell_sort_seq::ShellSortSequential ShellSortSequential(taskDataSeq);
  ASSERT_EQ(ShellSortSequential.validation(), true);
  ShellSortSequential.pre_processing();
  ShellSortSequential.run();
  ShellSortSequential.post_processing();

  std::sort(expected_result.begin(), expected_result.end());
  ASSERT_EQ(out, expected_result);
}
