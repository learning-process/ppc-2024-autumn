#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

static std::shared_ptr<ppc::core::TaskData> test_sobel_mk_taskdata(std::vector<uint8_t> &in, std::vector<uint8_t> &out,
                                                                   uint32_t width, uint32_t height) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs = {reinterpret_cast<uint8_t *>(in.data())};
  taskData->inputs_count = {width, height};

  taskData->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  taskData->outputs_count = {width, height};

  return taskData;
}

static void test_sobel_io(std::vector<uint8_t> &&in, uint32_t width, uint32_t height, std::vector<uint8_t> &out) {
  ASSERT_EQ(in.size(), width * height);

  auto taskData = test_sobel_mk_taskdata(in, out, width, height);

  koshkin_m_sobel_seq::TestTaskSequential testTaskSequential(taskData);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
}

static void test_sobel(std::vector<uint8_t> &&in, const std::vector<uint8_t> &ref, uint32_t width, uint32_t height) {
  std::vector<uint8_t> out(in.size());
  test_sobel_io(std::move(in), width, height, out);

  ASSERT_EQ(out, ref);
}

TEST(koshkin_m_sobel_seq, Image_1x1) { test_sobel({128}, {0}, 1, 1); }
TEST(koshkin_m_sobel_seq, Image_2x2) { test_sobel({32, 21, 132, 111}, {255, 255, 255, 255}, 2, 2); }
TEST(koshkin_m_sobel_seq, Image_2x3) { test_sobel({32, 21, 201, 231, 132, 111}, {255, 159, 255, 255, 243, 255}, 2, 3); }
TEST(koshkin_m_sobel_seq, Image_3x3) {
  test_sobel({32, 21, 61, 201, 231, 61, 132, 61, 111}, {255, 255, 255, 255, 255, 255, 255, 255, 255}, 3, 3);
}