#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/kovalchuk_a_odd_even_megre_sort/include/ops_seq.hpp"

using namespace kovalchuk_a_odd_even_seq;

std::vector<int> getRandomVectorrr(int sz, int min = -999, int max = 999) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

TEST(kovalchuk_a_odd_even_seq, Test_test) {
  std::vector<int> global_vector = {8, 2, 5, 10, 1, 7, 3, 12, 6, 11, 4, 9};
  std::vector<int> global_result(global_vector.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<int> reference_result = global_vector;
  std::sort(reference_result.begin(), reference_result.end());

  ASSERT_EQ(reference_result, global_result);
}

TEST(kovalchuk_a_odd_even_seq, Test_Vector_10) {
  const int count_elements = 10;
  std::vector<int> global_vector = getRandomVectorrr(count_elements);
  std::vector<int> global_result(count_elements, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(count_elements);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<int> reference_result = global_vector;
  std::sort(reference_result.begin(), reference_result.end());

  ASSERT_EQ(reference_result, global_result);
}

TEST(kovalchuk_a_odd_even_seq, Test_Vector_1) {
  const int count_elements = 1;
  std::vector<int> global_vector = getRandomVectorrr(count_elements, 0, 0);
  std::vector<int> global_result(count_elements, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(count_elements);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<int> reference_result = global_vector;
  std::sort(reference_result.begin(), reference_result.end());

  ASSERT_EQ(reference_result, global_result);
}
