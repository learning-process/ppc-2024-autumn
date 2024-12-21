#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sidorina_p_convex_hull_binary_image_mpi/include/ops_mpi.hpp"

std::vector<int> gen(int width, int height) {
  if (width <= 0 || height <= 0) {
    return {};
  }

  std::vector<int> image(width * height);
  for (int i = 0; i < width * height; ++i) {
    image[i] = rand() % 2;
  }

  return image;
}

TEST(sidorina_p_convex_hull_binary_image_mpi, Test_valid_vect_0) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int width = 3;
    const int height = 3;
    std::vector<int> image;
    std::vector<int> hull;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(sidorina_p_convex_hull_binary_image_mpi, Test_valid_width_0) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int width = 0;
    const int height = 2;
    std::vector<int> image = gen(width, height);
    std::vector<int> hull(width * height);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(sidorina_p_convex_hull_binary_image_mpi, Test_valid_height_0) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int width = 3;
    const int height = 0;
    std::vector<int> image = gen(width, height);
    std::vector<int> hull(width * height);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(sidorina_p_convex_hull_binary_image_mpi, Test_valid_neg) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int width = -2;
    const int height = -1;
    std::vector<int> image = gen(width, height);
    std::vector<int> hull(width * height);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(sidorina_p_convex_hull_binary_image_mpi, Test_valid_not_bin) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int width = 2;
    const int height = 4;
    std::vector<int> image = gen(width, height);
    std::vector<int> hull(width * height);

    image[2] = 2;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(sidorina_p_convex_hull_binary_image_mpi, Test_all_px_0) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;
  std::vector<int> image(width * height, 0);
  std::vector<int> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi TestTaskMPI(taskDataPar);
  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();

  if (world.rank() == 0) {
     ASSERT_EQ(image, hull);
  }
}