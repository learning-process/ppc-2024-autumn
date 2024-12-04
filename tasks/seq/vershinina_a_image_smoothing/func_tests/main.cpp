
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "seq/vershinina_a_image_smoothing/include/ops_seq.hpp"

std::vector<int> vershinina_a_image_smoothing::getRandomVector(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 255);
  std::vector<int> vec(rows * cols);
  int vec_size = vec.size();
  for (int i = 0; i < vec_size; i++) {
    vec[i] = distr(gen);
  }
  return vec;
}

TEST(vershinina_a_image_smoothing, Test1) {
  std::vector<int> in = {69,  60,  160, 247, 142, 68,  94,  16,  8,   33,  246, 78,  37,  22,  146, 253,
                         48,  168, 155, 228, 149, 110, 202, 111, 156, 222, 242, 48,  32,  59,  208, 39,
                         60,  16,  33,  97,  241, 42,  43,  122, 7,   194, 173, 188, 36,  1,   122, 175,
                         205, 238, 108, 39,  2,   76,  153, 184, 3,   128, 216, 139, 183, 109, 178, 23};
  std::vector<int> ref = {42,  96,  137, 151, 99, 84,  99,  127, 64,  105, 152, 160, 120, 107, 113, 137,
                          105, 142, 157, 135, 84, 107, 127, 159, 111, 122, 134, 136, 111, 120, 104, 120,
                          109, 122, 134, 121, 82, 87,  90,  118, 120, 114, 120, 101, 80,  79,  102, 133,
                          129, 141, 158, 120, 85, 95,  113, 139, 143, 149, 144, 114, 91,  116, 120, 134};
  std::vector<int> out(8 * 8);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(8);
  taskDataSeq->inputs_count.emplace_back(8);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  vershinina_a_image_smoothing::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}
TEST(vershinina_a_image_smoothing, Test2) {
  std::vector<int> in = {49,  10,  160, 227, 142, 78,  94,  16,  18,  33,  246, 74,  37,  22,  41,  253,
                         11,  168, 145, 228, 149, 110, 202, 211, 156, 232, 242, 48,  32,  59,  108, 39,
                         60,  16,  53,  97,  241, 42,  43,  122, 27,  19,  73,  198, 36,  1,   122, 175,
                         205, 248, 108, 39,  2,   76,  153, 194, 3,   128, 216, 139, 183, 109, 18,  50};
  std::vector<int> ref = {27,  86,  125, 147, 96, 69, 84,  101, 48,  93,  143, 156, 118, 97,  114, 136,
                          103, 139, 157, 133, 84, 84, 116, 142, 107, 120, 136, 137, 111, 109, 104, 120,
                          85,  97,  108, 113, 83, 76, 79,  101, 95,  89,  94,  94,  81,  79,  103, 134,
                          105, 114, 129, 110, 87, 77, 99,  118, 146, 151, 146, 114, 91,  90,  100, 103};
  std::vector<int> out(8 * 8);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(8);
  taskDataSeq->inputs_count.emplace_back(8);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  vershinina_a_image_smoothing::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}
