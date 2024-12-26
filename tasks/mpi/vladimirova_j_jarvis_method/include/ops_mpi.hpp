// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vladimirova_j_jarvis_method_mpi {

struct Point {
  int x, y;
  Point() {
    x = 0;
    y = 0;
  }
  Point(int _x, int _y) {
    x = _x;
    y = _y;
  }
  bool operator==(const Point& other) const { return (x == other.x) && (y == other.y); }
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Point> input_;
  std::vector<int> res_;
  size_t col = 0, row = 0;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<Point> local_input_;
  std::vector<int> res_;
  size_t col = 0, row = 0;
  boost::mpi::communicator world;
};

}  // namespace vladimirova_j_jarvis_method_mpi