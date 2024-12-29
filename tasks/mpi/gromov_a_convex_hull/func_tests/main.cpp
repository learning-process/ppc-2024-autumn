#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/gromov_a_convex_hull/include/ops_mpi.hpp"

namespace gromov_a_convex_hull_mpi {
std::vector<int> CreateCanvas(int height, int width, const std::string& figure) {
  std::vector<int> grid(width * height, 0);
  if (figure == "circle") {
    int centerX = width / 2;
    int centerY = height / 2;
    int radius = std::min(width, height) / 4;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int dx = x - centerX;
        int dy = y - centerY;
        if (dx * dx + dy * dy <= radius * radius) {
          grid[y * width + x] = 1;
        }
      }
    }
  } else if (figure == "square") {
    int size = std::min(width, height) / 2;
    int startX = (width - size) / 2;
    int startY = (height - size) / 2;
    for (int y = startY; y < startY + size; ++y) {
      for (int x = startX; x < startX + size; ++x) {
        grid[y * width + x] = 1;
      }
    }
  } else {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        grid[y * width + x] = dist(gen);
      }
    }
  }
  return grid;
}
}  // namespace gromov_a_convex_hull_mpi

TEST(gromov_a_convex_hull_mpi, Test_With_Zeroes) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;
  std::vector<int> grid(width * height, 0);
  std::vector<int32_t> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  gromov_a_convex_hull_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(grid, hull);
  }
}

TEST(gromov_a_convex_hull_mpi, Test_Circle) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;
  std::vector<int> grid = gromov_a_convex_hull_mpi::CreateCanvas(height, width, "circle");
  std::vector<int32_t> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  gromov_a_convex_hull_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NE(grid, hull);
  }
}

TEST(gromov_a_convex_hull_mpi, Test_Square) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;
  std::vector<int> grid = gromov_a_convex_hull_mpi::CreateCanvas(height, width, "square");
  std::vector<int32_t> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  gromov_a_convex_hull_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NE(grid, hull);
  }
}

TEST(gromov_a_convex_hull_mpi, Test_Big_Image) {
  boost::mpi::communicator world;
  const int width = 256;
  const int height = 256;
  std::vector<int> grid(width * height, 1);
  std::vector<int32_t> hull(width * height);

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  gromov_a_convex_hull_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(gromov_a_convex_hull_mpi, Test_Alternating_Pixels) {
  boost::mpi::communicator world;
  const int width = 4;
  const int height = 4;
  std::vector<int> grid(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      grid[y * width + x] = (x + y) % 2;
    }
  }
  std::vector<int32_t> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  gromov_a_convex_hull_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NE(grid, hull);
  }
}

TEST(gromov_a_convex_hull_mpi, Test_Big_Image2) {
  boost::mpi::communicator world;
  const int width = 300;
  const int height = 300;
  std::vector<int> grid(width * height, 1);
  std::vector<int32_t> hull(width * height);

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  gromov_a_convex_hull_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(gromov_a_convex_hull_mpi, Test_Big_Image3) {
  boost::mpi::communicator world;
  const int width = 400;
  const int height = 400;
  std::vector<int> grid(width * height, 1);
  std::vector<int32_t> hull(width * height);

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  gromov_a_convex_hull_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}