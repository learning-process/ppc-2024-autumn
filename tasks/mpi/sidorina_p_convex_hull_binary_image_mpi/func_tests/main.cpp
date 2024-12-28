#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sidorina_p_convex_hull_binary_image_mpi/include/ops_mpi.hpp"

namespace sidorina_p_convex_hull_binary_image_mpi {
std::vector<int> gen(int width, int height) {
  std::vector<int> image(width * height);
  for (int i = 0; i < width * height; ++i) {
    image[i] = rand() % 2;
  }

  return image;
}
}  // namespace sidorina_p_convex_hull_binary_image_mpi

using Params = std::tuple<int, int, std::vector<int>, std::vector<int>>;

class sidorina_p_convex_hull_binary_image_mpi_test : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(sidorina_p_convex_hull_binary_image_mpi_test, Test_image) {
  boost::mpi::communicator world;
  int width = std::get<0>(GetParam());
  int height = std::get<1>(GetParam());
  std::vector<int> image = std::get<2>(GetParam());
  std::vector<int> hull(width * height);
  std::vector<int> ref = std::get<3>(GetParam());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(image.size());
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
    ASSERT_EQ(ref, hull);
  }
}

INSTANTIATE_TEST_SUITE_P(
    sidorina_p_convex_hull_binary_image_mpi_test_val, sidorina_p_convex_hull_binary_image_mpi_test,
    ::testing::Values(Params(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}),
                      Params(3, 3, {0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}),
                      Params(3, 3, {0, 1, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 1, 0, 1, 1, 0, 1, 1}),
                      Params(3, 4, {0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0}, {1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0}),
                      Params(4, 4, {0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0},
                             {1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0}),
                      Params(10, 10, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                                      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                                      0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1},
                             {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
                      Params(5, 5, {0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0},
                             {0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0})));

using Params_val = std::tuple<int, int, std::vector<int>>;

class sidorina_p_convex_hull_binary_image_mpi_test_val : public ::testing::TestWithParam<Params_val> {
 protected:
};

TEST_P(sidorina_p_convex_hull_binary_image_mpi_test_val, Test_validation) {
  boost::mpi::communicator world;
  int width = std::get<0>(GetParam());
  int height = std::get<1>(GetParam());
  std::vector<int> image = std::get<2>(GetParam());
  std::vector<int> hull(width * height);
  std::vector<int> ref(width * height, 0);

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(image.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    sidorina_p_convex_hull_binary_image_mpi::ConvexHullBinImgMpi TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

INSTANTIATE_TEST_SUITE_P(sidorina_p_convex_hull_binary_image_mpi_test_val,
                         sidorina_p_convex_hull_binary_image_mpi_test_val,
                         ::testing::Values(Params_val(0, 6, {1}), Params_val(3, 0, {1}), Params_val(5, 5, {2}),
                                           Params_val(5, 5, {})));