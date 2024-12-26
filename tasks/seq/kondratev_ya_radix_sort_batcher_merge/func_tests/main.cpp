// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/kondratev_ya_radix_sort_batcher_merge/include/ops_seq.hpp"
namespace kondratev_ya_radix_sort_batcher_merge_seq {
std::vector<double> getRandomVector(uint32_t size) {
  std::srand(std::time(nullptr));
  std::vector<double> vec(size);

  double lower_bound = -10000;
  double upper_bound = 10000;
  for (uint32_t i = 0; i < size; i++) {
    vec[i] = lower_bound + std::rand() / (double)RAND_MAX * (upper_bound - lower_bound);
  }
  return vec;
}
}  // namespace kondratev_ya_radix_sort_batcher_merge_seq

TEST(kondratev_ya_radix_sort_batcher_merge_seq, basic) {
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  uint32_t size = 12;
  in.assign({8, 2, 5, 10, 1, 7, 3, 12, 6, 11, 4, 9});
  out.resize(size);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (uint32_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_seq, empty) {
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
}

TEST(kondratev_ya_radix_sort_batcher_merge_seq, scalar) {
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  uint32_t size = 1;
  in = kondratev_ya_radix_sort_batcher_merge_seq::getRandomVector(size);
  out.resize(size);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (uint32_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_seq, prime) {
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  uint32_t size = 239;
  in = kondratev_ya_radix_sort_batcher_merge_seq::getRandomVector(size);
  out.resize(size);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (uint32_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_seq, power2) {
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  uint32_t size = 256;
  in = kondratev_ya_radix_sort_batcher_merge_seq::getRandomVector(size);
  out.resize(size);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (uint32_t i = 1; i < out.size(); i++) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}
