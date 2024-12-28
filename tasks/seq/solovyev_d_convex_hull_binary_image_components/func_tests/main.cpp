#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/solovyev_d_convex_hull_binary_image_components/include/header.hpp"

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_Wrong_Input_Dimensions) {
  int dimX = 1;
  int dimY = 1;
  // Create data
  std::vector<int> in = {1, 0, 1, 1, 1, 0, 1, 0};
  std::vector<std::vector<int>> expected = {};
  std::vector<std::vector<int>> out = {};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), false);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_Empty) {
  int dimX = 0;
  int dimY = 0;
  // Create data
  std::vector<int> in = {};
  std::vector<std::vector<int>> expected = {};
  std::vector<std::vector<int>> out = {};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_OnlyBackground) {
  int dimX = 5;
  int dimY = 5;
  // Create data
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<std::vector<int>> expected = {};
  std::vector<std::vector<int>> out = {};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out[i].size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_1x1) {
  int dimX = 1;
  int dimY = 1;
  // Create data
  std::vector<int> in = {1};
  std::vector<std::vector<int>> expected = {{0, 0}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_2x2) {
  int dimX = 2;
  int dimY = 2;
  // Create data
  std::vector<int> in = {1, 1, 1, 1};
  std::vector<std::vector<int>> expected = {{1, 1, 1, 0, 0, 0, 0, 1}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_3x3) {
  int dimX = 3;
  int dimY = 3;
  // Create data
  std::vector<int> in = {1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<std::vector<int>> expected = {{2, 0, 0, 0}, {2, 2, 0, 2}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_5x5) {
  int dimX = 5;
  int dimY = 5;
  // Create data
  std::vector<int> in = {0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0};
  std::vector<std::vector<int>> expected = {{3, 3, 4, 2, 4, 0, 3, 0, 1, 1}, {1, 4, 1, 3, 0, 3, 0, 4}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_4x6) {
  int dimX = 4;
  int dimY = 6;
  // Create data
  std::vector<int> in = {0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1};
  std::vector<std::vector<int>> expected = {{3, 3, 3, 0, 1, 1}, {3, 5, 1, 3, 0, 3, 0, 5}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out[i].size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(solovyev_d_convex_hull_binary_image_components_seq, Test_21x8) {
  int dimX = 21;
  int dimY = 8;
  // Create data
  std::vector<int> in = {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                         0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                         0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                         0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
                         0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                         0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0};
  std::vector<std::vector<int>> expected = {{6, 7, 6, 3, 3, 0, 0, 3, 0, 7},
                                            {12, 7, 13, 6, 13, 1, 12, 0, 9, 0, 8, 1, 8, 6, 9, 7},
                                            {18, 7, 20, 4, 20, 3, 18, 0, 17, 0, 15, 3, 15, 4, 17, 7}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0), std::vector<int>(expected[1].size(), 0),
                                       std::vector<int>(expected[2].size(), 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
  taskDataSeq->inputs_count.emplace_back(in.size());
  for (size_t i = 0; i < out.size(); i++) {
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    taskDataSeq->outputs_count.emplace_back(out[i].size());
  }

  // Create Task
  solovyev_d_convex_hull_binary_image_components_seq::ConvexHullBinaryImageComponentsSequential
      ConvexHullBinaryImageComponentsSequential(taskDataSeq);
  ASSERT_EQ(ConvexHullBinaryImageComponentsSequential.validation(), true);
  ConvexHullBinaryImageComponentsSequential.pre_processing();
  ConvexHullBinaryImageComponentsSequential.run();
  ConvexHullBinaryImageComponentsSequential.post_processing();
  ASSERT_EQ(expected, out);
}