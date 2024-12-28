#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

static pikarychev_i_gauss3_vertical_mpi::Image CreateRandomImage(size_t width, size_t height) {
  auto img = pikarychev_i_gauss3_vertical_mpi::Image::alloc(width, height);

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(0, 255);
  std::generate(img.data.begin(), img.data.end(), [&] {
    return pikarychev_i_gauss3_vertical_mpi::Image::Pixel{
        .r = (uint8_t)distrib(gen), .g = (uint8_t)distrib(gen), .b = (uint8_t)distrib(gen)};
  });

  return img;
}

static void g3x3t(size_t width, size_t height, pikarychev_i_gauss3_vertical_mpi::Kernel3x3 &&kernel) {
  boost::mpi::communicator world;

  pikarychev_i_gauss3_vertical_mpi::Image in;
  pikarychev_i_gauss3_vertical_mpi::Image out;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = CreateRandomImage(width, height);
    out = pikarychev_i_gauss3_vertical_mpi::Image::alloc(in.width, in.height);
  }
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&kernel));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  pikarychev_i_gauss3_vertical_mpi::TaskPar task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    auto ref = pikarychev_i_gauss3_vertical_mpi::Image::alloc(in.width, in.height);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>(*taskData);
    taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(&ref)};

    pikarychev_i_gauss3_vertical_mpi::TaskSeq taskSeq(taskDataSeq);
    ASSERT_EQ(taskSeq.validation(), true);
    taskSeq.pre_processing();
    taskSeq.run();
    taskSeq.post_processing();

    ASSERT_EQ(out, ref);
  }
}

TEST(pikarychev_i_gauss3_vertical_mpi, govno) { g3x3t(3, 1, {-1, -2, -1, 0, 0, 0, 1, 2, 1}); }

TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_1x1) { g3x3t(1, 1, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_2x2) { g3x3t(2, 2, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_2x3) { g3x3t(2, 3, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_2x4) { g3x3t(2, 4, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_3x2) { g3x3t(3, 2, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_4x4) { g3x3t(4, 4, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_7x41) { g3x3t(7, 41, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, 010Kernel_NoChanges_63x31) { g3x3t(63, 31, {0, 0, 0, 0, 1, 0, 0, 0, 0}); }

TEST(pikarychev_i_gauss3_vertical_mpi, SobelX_1x1) { g3x3t(1, 1, {-1, 0, 1, -2, 0, 2, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelX_2x2) { g3x3t(2, 2, {-1, 0, 1, -2, 0, 2, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelX_2x3) { g3x3t(2, 3, {-1, 0, 1, -2, 0, 2, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelX_3x2) { g3x3t(3, 2, {-1, 0, 1, -2, 0, 2, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelX_7x41) { g3x3t(7, 41, {-1, 0, 1, -2, 0, 2, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelX_63x31) { g3x3t(63, 31, {-1, 0, 1, -2, 0, 2, -1, 0, 1}); }

TEST(pikarychev_i_gauss3_vertical_mpi, SobelY_1x1) { g3x3t(1, 1, {-1, -2, -1, 0, 0, 0, 1, 2, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelY_2x2) { g3x3t(2, 2, {-1, -2, -1, 0, 0, 0, 1, 2, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelY_2x3) { g3x3t(2, 3, {-1, -2, -1, 0, 0, 0, 1, 2, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelY_3x2) { g3x3t(3, 2, {-1, -2, -1, 0, 0, 0, 1, 2, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelY_7x41) { g3x3t(7, 41, {-1, -2, -1, 0, 0, 0, 1, 2, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, SobelY_63x31) { g3x3t(63, 31, {-1, -2, -1, 0, 0, 0, 1, 2, 1}); }

TEST(pikarychev_i_gauss3_vertical_mpi, Sharpness_1x1) { g3x3t(1, 1, {0, -1, 0, -1, 5, -1, 0, -1, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, Sharpness_2x2) { g3x3t(2, 2, {0, -1, 0, -1, 5, -1, 0, -1, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, Sharpness_2x3) { g3x3t(2, 3, {0, -1, 0, -1, 5, -1, 0, -1, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, Sharpness_3x2) { g3x3t(3, 2, {0, -1, 0, -1, 5, -1, 0, -1, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, Sharpness_7x41) { g3x3t(7, 41, {0, -1, 0, -1, 5, -1, 0, -1, 0}); }
TEST(pikarychev_i_gauss3_vertical_mpi, Sharpness_63x31) { g3x3t(63, 31, {0, -1, 0, -1, 5, -1, 0, -1, 0}); }

TEST(pikarychev_i_gauss3_vertical_mpi, PrewittX_1x1) { g3x3t(1, 1, {-1, 0, 1, -1, 0, 1, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittX_2x2) { g3x3t(2, 2, {-1, 0, 1, -1, 0, 1, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittX_2x3) { g3x3t(2, 3, {-1, 0, 1, -1, 0, 1, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittX_3x2) { g3x3t(3, 2, {-1, 0, 1, -1, 0, 1, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittX_7x41) { g3x3t(7, 41, {-1, 0, 1, -1, 0, 1, -1, 0, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittX_63x31) { g3x3t(63, 31, {-1, 0, 1, -1, 0, 1, -1, 0, 1}); }

TEST(pikarychev_i_gauss3_vertical_mpi, PrewittY_1x1) { g3x3t(1, 1, {-1, -1, -1, 0, 0, 0, 1, 1, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittY_2x2) { g3x3t(2, 2, {-1, -1, -1, 0, 0, 0, 1, 1, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittY_2x3) { g3x3t(2, 3, {-1, -1, -1, 0, 0, 0, 1, 1, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittY_3x2) { g3x3t(3, 2, {-1, -1, -1, 0, 0, 0, 1, 1, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittY_7x41) { g3x3t(7, 41, {-1, -1, -1, 0, 0, 0, 1, 1, 1}); }
TEST(pikarychev_i_gauss3_vertical_mpi, PrewittY_63x31) { g3x3t(63, 31, {-1, -1, -1, 0, 0, 0, 1, 1, 1}); }

TEST(pikarychev_i_gauss3_vertical_mpi, Gauss_1x1) {
  g3x3t(1, 1, {1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16.});
}
TEST(pikarychev_i_gauss3_vertical_mpi, Gauss_2x2) {
  g3x3t(2, 2, {1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16.});
}
TEST(pikarychev_i_gauss3_vertical_mpi, Gauss_2x3) {
  g3x3t(2, 3, {1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16.});
}
TEST(pikarychev_i_gauss3_vertical_mpi, Gauss_3x2) {
  g3x3t(3, 2, {1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16.});
}
TEST(pikarychev_i_gauss3_vertical_mpi, Gauss_7x41) {
  g3x3t(7, 41, {1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16.});
}
TEST(pikarychev_i_gauss3_vertical_mpi, Gauss_63x31) {
  g3x3t(63, 31, {1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16.});
}