#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

static std::vector<uint8_t> make_img(size_t width, size_t height) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(0, 255);
  std::vector<uint8_t> vec(width * height);
  for (size_t i = 0; i < width * height; i++) {
    vec[i] = distrib(gen);
  }
  return vec;
}

static std::shared_ptr<ppc::core::TaskData> test_sobel_mk_taskdata(std::vector<uint8_t> &in, std::vector<uint8_t> &out,
                                                                   uint32_t width, uint32_t height) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  if (boost::mpi::communicator{}.rank() == 0) {
    taskData->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskData->inputs_count = {width, height};

    taskData->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskData->outputs_count = {width, height};
  }

  return taskData;
}

static void test_sobel_io(std::vector<uint8_t> &&in, uint32_t width, uint32_t height, std::vector<uint8_t> &out) {
  ASSERT_EQ(in.size(), width * height);

  auto taskData = test_sobel_mk_taskdata(in, out, width, height);

  koshkin_m_sobel_mpi::TestTaskParallel testTaskParallel(taskData);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
}

static void test_sobel(uint32_t width, uint32_t height) {
  std::vector<uint8_t> in = make_img(width, height);
  std::vector<uint8_t> out;
  if (boost::mpi::communicator{}.rank() == 0) {
    out.resize(in.size());
  }
  test_sobel_io(std::move(in), width, height, out);

  if (boost::mpi::communicator{}.rank() == 0) {
    std::vector<uint8_t> ref(in.size());

    koshkin_m_sobel_mpi::TestTaskSequential testTaskSequential(test_sobel_mk_taskdata(in, ref, width, height));
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out, ref);
  }
}

TEST(koshkin_m_sobel_mpi, Image_Random_1x1) { test_sobel(1, 1); }
TEST(koshkin_m_sobel_mpi, Image_Random_2x2) { test_sobel(2, 2); }
TEST(koshkin_m_sobel_mpi, Image_Random_2x3) { test_sobel(2, 3); }
TEST(koshkin_m_sobel_mpi, Image_Random_3x3) { test_sobel(3, 3); }
TEST(koshkin_m_sobel_mpi, Image_Random_3x7) { test_sobel(3, 7); }
TEST(koshkin_m_sobel_mpi, Image_Random_7x3) { test_sobel(7, 3); }
TEST(koshkin_m_sobel_mpi, Image_Random_7x13) { test_sobel(7, 13); }
TEST(koshkin_m_sobel_mpi, Image_Random_13x7) { test_sobel(13, 7); }
TEST(koshkin_m_sobel_mpi, Image_Random_17x13) { test_sobel(17, 13); }
TEST(koshkin_m_sobel_mpi, Image_Random_1x13) { test_sobel(1, 13); }
TEST(koshkin_m_sobel_mpi, Image_Random_17x17) { test_sobel(17, 17); }