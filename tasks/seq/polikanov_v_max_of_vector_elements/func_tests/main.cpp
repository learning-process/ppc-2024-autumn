#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/polikanov_v_max_of_vector_elements/include/ops_seq.hpp"
namespace polikanov_v {
std::vector<int> getRandomVector(int sz) {
  int lower = -100;
  int upper = 100;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = lower + gen() % (upper - lower + 1);
  }
  return vec;
}
}  // namespace polikanov_v

TEST(polikanov_v_max_of_vector_elements_seq, Test_Seq_1) {
  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  polikanov_v_max_of_vector_elements::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(6, out[0]);
}

TEST(polikanov_v_max_of_vector_elements_seq, Test_Seq_2) {
  // Create data
  std::vector<int> in = polikanov_v::getRandomVector(10);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  polikanov_v_max_of_vector_elements::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  int max = INT_MIN;
  for (size_t i = 0; i < in.size(); ++i) {
    max = std::max(max, in[i]);
  }
  ASSERT_EQ(max, out[0]);
}
