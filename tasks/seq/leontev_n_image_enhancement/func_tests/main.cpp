#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/leontev_n_image_enhancement/include/ops_seq.hpp"

namespace leontev_n_image_enhancement_seq {
std::vector<int> getRandomImage(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace leontev_n_image_enhancement_seq

template <class InOutType>
void taskEmplacement(std::shared_ptr<ppc::core::TaskData> &taskDataPar, std::vector<InOutType> &global_vec,
                     std::vector<InOutType> &global_sum) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
  taskDataPar->outputs_count.emplace_back(global_sum.size());
}

TEST(leontev_n_image_enhancement_seq, test_image_1) {
  const int vector_size = 9;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  leontev_n_image_enhancement_seq::ImgEnhancementSequential imgEnhancementSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {15, 50, 45, 101, 92, 79, 0, 0, 15};
  std::vector<int> out_vec(vector_size, 0);

  // Create TaskData

  taskEmplacement(taskDataSeq, in_vec, out_vec);

  // Create Task
  ASSERT_EQ(imgEnhancementSequential.validation(), true);
  imgEnhancementSequential.pre_processing();
  imgEnhancementSequential.run();
  imgEnhancementSequential.post_processing();

  ASSERT_EQ(out_vec, std::vector<int>({38, 129, 116, 255, 255, 223, 0, 0, 0}));
}

TEST(leontev_n_image_enhancement_seq, test_image_2) {
  const int vector_size = 27;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  leontev_n_image_enhancement_seq::ImgEnhancementSequential imgEnhancementSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec(27);
  for (size_t i = 0; i < in_vec.size(); i++) {
    in_vec[i] = i;
  }
  std::vector<int> out_vec(vector_size, 0);

  // Create TaskData

  taskEmplacement(taskDataSeq, in_vec, out_vec);

  // Create Task
  ASSERT_EQ(imgEnhancementSequential.validation(), true);
  imgEnhancementSequential.pre_processing();
  imgEnhancementSequential.run();
  imgEnhancementSequential.post_processing();

  ASSERT_EQ(out_vec, std::vector<int>({0,   0,   0,   23,  31,  38,  54,  63,  72,  85,  95,  104, 117, 127,
                                       136, 149, 159, 168, 180, 191, 201, 212, 223, 233, 244, 255, 255}));
}

TEST(leontev_n_image_enhancement_seq, test_incorrect_size) {
  const int vector_size = 8;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  leontev_n_image_enhancement_seq::ImgEnhancementSequential imgEnhancementSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {15, 50, 45, 101, 92, 79, 0, 0};
  std::vector<int> out_vec(vector_size, 0);

  // Create TaskData

  taskEmplacement(taskDataSeq, in_vec, out_vec);

  // Create Task
  ASSERT_EQ(imgEnhancementSequential.validation(), false);
}

TEST(leontev_n_image_enhancement_seq, test_incorrect_color_range) {
  const int vector_size = 12;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  leontev_n_image_enhancement_seq::ImgEnhancementSequential imgEnhancementSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {631, -2, 45, 101, 92, 79, 0, 0, 300, 255, 10, 15};
  std::vector<int> out_vec(vector_size, 0);

  // Create TaskData

  taskEmplacement(taskDataSeq, in_vec, out_vec);

  // Create Task
  ASSERT_EQ(imgEnhancementSequential.validation(), false);
}

TEST(leontev_n_image_enhancement_seq, test_empty_image) {
  const int vector_size = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  leontev_n_image_enhancement_seq::ImgEnhancementSequential imgEnhancementSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {};
  std::vector<int> out_vec(vector_size, 0);

  // Create TaskData

  taskEmplacement(taskDataSeq, in_vec, out_vec);

  // Create Task
  ASSERT_EQ(imgEnhancementSequential.validation(), false);
}