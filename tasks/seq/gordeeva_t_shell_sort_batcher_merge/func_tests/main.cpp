#include <gtest/gtest.h>

#include <vector>

#include "seq/gordeeva_t_shell_sort_batcher_merge/include/ops_seq.hpp"

static std::vector<int> rand_vec(int size, int down = -100, int upp = 100) {
  std::vector<int> v(size);
  for (auto &number : v) number = down + (std::rand() % (upp - down + 1));
  return v;
}

TEST(gordeeva_t_shell_sort_batcher_merge_seq, Shell_sort_Zero_Value) {
  const int sz_vec = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> vect = rand_vec(sz_vec);

  taskDataSeq->inputs_count.emplace_back(sz_vec);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));

  std::vector<int> res(sz_vec, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(gordeeva_t_shell_sort_batcher_merge_seq, Shell_sort_Empty_Output) {
  const int sz_vec = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> vect = rand_vec(sz_vec);

  taskDataSeq->inputs_count.emplace_back(sz_vec);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(gordeeva_t_shell_sort_batcher_merge_seq, Shell_sort_100_with_random) {
  const int sz_vec = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> vect = rand_vec(sz_vec);

  taskDataSeq->inputs_count.emplace_back(sz_vec);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));

  std::vector<int> res(sz_vec, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < res.size(); i++) {
    ASSERT_LE(res[i - 1], res[i]);
  }
}

TEST(gordeeva_t_shell_sort_batcher_merge_seq, Shell_sort_1000_with_random) {
  const int sz_vec = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> vect = rand_vec(sz_vec);

  taskDataSeq->inputs_count.emplace_back(sz_vec);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));

  std::vector<int> res(sz_vec, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < res.size(); i++) {
    ASSERT_LE(res[i - 1], res[i]);
  }
}

TEST(gordeeva_t_shell_sort_batcher_merge_seq, Shell_sort_5000_with_random) {
  const int sz_vec = 5000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> vect = rand_vec(sz_vec);

  taskDataSeq->inputs_count.emplace_back(sz_vec);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));

  std::vector<int> res(sz_vec, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 1; i < res.size(); i++) {
    ASSERT_LE(res[i - 1], res[i]);
  }
}
