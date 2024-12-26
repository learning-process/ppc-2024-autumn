#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gromov_a_convex_hull_mpi {

struct Point {
  int x, y;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> grid;
  std::vector<std::vector<Point>> components;
  int width{};
  int height{};
  int size{};
  boost::mpi::communicator world;
};

}  // namespace gromov_a_convex_hull_mpi