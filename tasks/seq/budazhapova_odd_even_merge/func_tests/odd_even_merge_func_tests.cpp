#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "seq/budazhapova_odd_even_merge/include/odd_even_merge.hpp"

namespace budazhapova_betcher_odd_even_merge_seq {
std::vector<int> generateRandomVector(int size, int minValue, int maxValue) {
  std::vector<int> randomVector;
  randomVector.reserve(size);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < size; ++i) {
    int randomNum = std::rand() % (maxValue - minValue + 1) + minValue;
    randomVector.push_back(randomNum);
  }
  return randomVector;
}
}  // namespace budazhapova_betcher_odd_even_merge_seq
TEST(budazhapova_betcher_odd_even_merge_seq, ordinary_test) {
  std::vector<int> input_vector = {34, 12, 5, 78, 23, 45, 67, 89, 10, 2, 56, 43, 91, 15, 30};
  std::vector<int> out(15, 0);
  std::vector<int> sorted_vector = {2, 5, 10, 12, 15, 23, 30, 34, 43, 45, 56, 67, 78, 89, 91};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  budazhapova_betcher_odd_even_merge_seq::MergeSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, sorted_vector);
}

TEST(budazhapova_betcher_odd_even_merge_seq, random_vector_test) {
  std::vector<int> input_vector = generateRandomVector(100, 5, 100);
  std::vector<int> out(100, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  budazhapova_betcher_odd_even_merge_seq::MergeSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
}

TEST(budazhapova_betcher_odd_even_merge_seq, validation_test) {
  std::vector<int> input_vector = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  budazhapova_betcher_odd_even_merge_seq::MergeSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}
