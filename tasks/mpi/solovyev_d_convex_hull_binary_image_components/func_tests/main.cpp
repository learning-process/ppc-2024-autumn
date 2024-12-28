#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/solovyev_d_convex_hull_binary_image_components/include/header.hpp"

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_Wrong_Input_Dimensions) {
  boost::mpi::communicator world;
  int dimX = 1;
  int dimY = 1;

  std::vector<int> in = {1, 0, 1, 1, 1, 0, 1, 0};
  std::vector<std::vector<int>> expected = {};
  std::vector<std::vector<int>> out = {};

  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), false);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_Empty) {
  boost::mpi::communicator world;
  int dimX = 0;
  int dimY = 0;
  std::vector<int> in = {};
  std::vector<std::vector<int>> expected = {};
  std::vector<std::vector<int>> out = {};

  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_OnlyBackground) {
  boost::mpi::communicator world;
  int dimX = 5;
  int dimY = 5;
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<std::vector<int>> expected = {};
  std::vector<std::vector<int>> out = {};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out[i].size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_1x1) {
  boost::mpi::communicator world;
  int dimX = 1;
  int dimY = 1;
  std::vector<int> in = {1};
  std::vector<std::vector<int>> expected = {{0, 0}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0)};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_2x2) {
  boost::mpi::communicator world;
  int dimX = 2;
  int dimY = 2;
  std::vector<int> in = {1, 1, 1, 1};
  std::vector<std::vector<int>> expected = {{1, 1, 1, 0, 0, 0, 0, 1}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0)};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_3x3) {
  boost::mpi::communicator world;
  int dimX = 3;
  int dimY = 3;
  std::vector<int> in = {1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<std::vector<int>> expected = {{2, 0, 0, 0}, {2, 2, 0, 2}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_5x5) {
  boost::mpi::communicator world;
  int dimX = 5;
  int dimY = 5;
  std::vector<int> in = {0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0};
  std::vector<std::vector<int>> expected = {{3, 3, 4, 2, 4, 0, 3, 0, 1, 1}, {1, 4, 1, 3, 0, 3, 0, 4}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_4x6) {
  boost::mpi::communicator world;
  int dimX = 4;
  int dimY = 6;
  std::vector<int> in = {0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1};
  std::vector<std::vector<int>> expected = {{3, 3, 3, 0, 1, 1}, {3, 5, 1, 3, 0, 3, 0, 5}};
  std::vector<std::vector<int>> out = {std::vector<int>(expected[0].size(), 0),
                                       std::vector<int>(expected[1].size(), 0)};
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out[i].size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(solovyev_d_convex_hull_binary_image_components_mpi, Test_21x8) {
  boost::mpi::communicator world;
  int dimX = 21;
  int dimY = 8;
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
  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimX));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimY));
    taskDataPar->inputs_count.emplace_back(in.size());
    for (size_t i = 0; i < out.size(); i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
      taskDataPar->outputs_count.emplace_back(out[i].size());
    }
  }
  // Create Task
  solovyev_d_convex_hull_binary_image_components_mpi::ConvexHullBinaryImageComponentsMPI
      ConvexHullBinaryImageComponentsMPI(taskDataPar);
  ASSERT_EQ(ConvexHullBinaryImageComponentsMPI.validation(), true);
  ConvexHullBinaryImageComponentsMPI.pre_processing();
  ConvexHullBinaryImageComponentsMPI.run();
  ConvexHullBinaryImageComponentsMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}