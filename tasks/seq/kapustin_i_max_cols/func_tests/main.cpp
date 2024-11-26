#include <gtest/gtest.h>

#include <vector>

#include "seq/kapustin_i_max_cols/include/avg_seq.hpp"
namespace kapustin_i_max_column_task_seq {
std::vector<int> getRandomVectorForSeq(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    int val = gen() % 200000 - 100000;
    vec[i] = val;
  }
  return vec;
}
}  // namespace kapustin_i_max_column_task_seq

TEST(kapustin_i_max_column_task_seq, test_square_M_3_3) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> expected_out = {7, 8, 9};
  std::vector<int> out(3);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int columns = 3;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&columns));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->inputs_count.emplace_back(columns);

  kapustin_i_max_column_task_seq::MaxColumnTaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out, expected_out);
}

TEST(kapustin_i_max_column_task_seq, test_rect_M_2_4) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> expected_out = {5, 6, 7, 8};
  std::vector<int> out(4);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int columns = 4;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&columns));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->inputs_count.emplace_back(columns);

  kapustin_i_max_column_task_seq::MaxColumnTaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(out, expected_out);
}

TEST(kapustin_i_max_column_task_seq, test_big_square_M_6_6) {
  std::vector<int> in = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                         19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  std::vector<int> expected_out = {31, 32, 33, 34, 35, 36};
  std::vector<int> out(6);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int columns = 6;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&columns));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->inputs_count.emplace_back(columns);

  kapustin_i_max_column_task_seq::MaxColumnTaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(out, expected_out);
}

TEST(kapustin_i_max_column_task_seq, empty_M) {
  std::vector<int> in = {};
  std::vector<int> expected_out = {};
  std::vector<int> out;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int columns = 0;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&columns));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->inputs_count.emplace_back(columns);

  kapustin_i_max_column_task_seq::MaxColumnTaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), false);
}

TEST(kapustin_i_max_column_task_seq, identity_M) {
  std::vector<int> in = {1};
  std::vector<int> expected_out = {1};
  std::vector<int> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int columns = 1;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&columns));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->inputs_count.emplace_back(columns);

  kapustin_i_max_column_task_seq::MaxColumnTaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(out, expected_out);
}

TEST(kapustin_i_max_column_task_seq, InvalidInputCountSize) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count = {1};
  kapustin_i_max_column_task_seq::MaxColumnTaskSequential task(taskData);
  ASSERT_FALSE(task.validation());
}
TEST(kapustin_i_max_column_task_seq, ZeroColumnOrRowCount) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count = {0, 0};
  kapustin_i_max_column_task_seq::MaxColumnTaskSequential task(taskData);
  ASSERT_FALSE(task.validation());
}
TEST(kapustin_i_max_column_task_seq, test_square_M_random) {
  int rows = 3;
  int columns = 3;
  std::vector<int> in = kapustin_i_max_column_task_seq::getRandomVectorForSeq(rows * columns);
  std::vector<int> expected_out(columns);
  for (int col = 0; col < columns; ++col) {
    int max_val = in[col];
    for (int row = 1; row < rows; ++row) {
      int idx = row * columns + col;
      if (in[idx] > max_val) {
        max_val = in[idx];
      }
    }
    expected_out[col] = max_val;
  }
  std::vector<int> out(columns);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&columns));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  kapustin_i_max_column_task_seq::MaxColumnTaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  ASSERT_EQ(out, expected_out);
}
