#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

static void g3x3t(pikarychev_i_gauss3_vertical_seq::Image &&img, pikarychev_i_gauss3_vertical_seq::Kernel3x3 &&kernel,
                  const pikarychev_i_gauss3_vertical_seq::Image &ref) {
  auto out = pikarychev_i_gauss3_vertical_seq::Image::alloc(img.width, img.height);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&img));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&kernel));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  pikarychev_i_gauss3_vertical_seq::TaskSeq task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_EQ(out, ref);
}

TEST(pikarychev_i_gauss3_vertical_seq, 010Kernel_NoChanges_1x1) {
  g3x3t({1, 1, {{21, 24, 81}}}, {0, 0, 0, 0, 1, 0, 0, 0, 0}, {1, 1, {{21, 24, 81}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, 010Kernel_NoChanges_2x2) {
  g3x3t({2, 2, {{21, 24, 81}, {232, 31, 39}, {22, 31, 39}, {232, 77, 39}}}, {0, 0, 0, 0, 1, 0, 0, 0, 0},
        {2, 2, {{21, 24, 81}, {232, 31, 39}, {22, 31, 39}, {232, 77, 39}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, 010Kernel_NoChanges_2x4) {
  g3x3t({2,
         4,
         {{171, 178, 240},
          {205, 184, 121},
          {34, 226, 230},
          {206, 1, 79},
          {98, 26, 203},
          {17, 89, 174},
          {53, 211, 86},
          {186, 193, 61}}},
        {0, 0, 0, 0, 1, 0, 0, 0, 0},
        {2,
         4,
         {{171, 178, 240},
          {205, 184, 121},
          {34, 226, 230},
          {206, 1, 79},
          {98, 26, 203},
          {17, 89, 174},
          {53, 211, 86},
          {186, 193, 61}}});
}

TEST(pikarychev_i_gauss3_vertical_seq, SobelX_1x1) {
  g3x3t({1, 1, {{21, 24, 81}}}, {-1, 0, 1, -2, 0, 2, -1, 0, 1}, {1, 1, {{0, 0, 0}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, SobelY_1x1) {
  g3x3t({1, 1, {{21, 24, 81}}}, {-1, -2, -1, 0, 0, 0, 1, 2, 1}, {1, 1, {{0, 0, 0}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, Sharpness_1x1) {
  g3x3t({1, 1, {{21, 24, 81}}}, {0, -1, 0, -1, 5, -1, 0, -1, 0}, {1, 1, {{105, 120, 149}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, PrewittX_1x1) {
  g3x3t({1, 1, {{21, 24, 81}}}, {-1, 0, 1, -1, 0, 1, -1, 0, 1}, {1, 1, {{0, 0, 0}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, PrewittY_1x1) {
  g3x3t({1, 1, {{21, 24, 81}}}, {-1, -1, -1, 0, 0, 0, 1, 1, 1}, {1, 1, {{0, 0, 0}}});
}

TEST(pikarychev_i_gauss3_vertical_seq, SobelX_2x2) {
  g3x3t({2, 2, {{21, 24, 81}, {232, 31, 39}, {22, 31, 39}, {232, 77, 39}}}, {-1, 0, 1, -2, 0, 2, -1, 0, 1},
        {2, 2, {{20, 139, 117}, {230, 185, 117}, {238, 177, 55}, {27, 170, 97}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, SobelY_2x2) {
  g3x3t({2, 2, {{21, 24, 81}, {232, 31, 39}, {22, 31, 39}, {232, 77, 39}}}, {-1, -2, -1, 0, 0, 0, 1, 2, 1},
        {2, 2, {{184, 139, 117}, {192, 177, 55}, {184, 185, 117}, {191, 170, 97}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, Sharpness_2x2) {
  g3x3t({2, 2, {{21, 24, 81}, {232, 31, 39}, {22, 31, 39}, {232, 77, 39}}}, {0, -1, 0, -1, 5, -1, 0, -1, 0},
        {2, 2, {{107, 58, 71}, {139, 54, 75}, {113, 54, 75}, {138, 67, 117}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, PrewittX_2x2) {
  g3x3t({2, 2, {{21, 24, 81}, {232, 31, 39}, {22, 31, 39}, {232, 77, 39}}}, {-1, 0, 1, -1, 0, 1, -1, 0, 1},
        {2, 2, {{254, 108, 78}, {254, 108, 78}, {3, 201, 136}, {3, 201, 136}}});
}
TEST(pikarychev_i_gauss3_vertical_seq, PrewittY_2x2) {
  g3x3t({2, 2, {{21, 24, 81}, {232, 31, 39}, {22, 31, 39}, {232, 77, 39}}}, {-1, -1, -1, 0, 0, 0, 1, 1, 1},
        {2, 2, {{208, 108, 78}, {213, 201, 136}, {208, 108, 78}, {213, 201, 136}}});
}