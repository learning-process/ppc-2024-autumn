#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_bellman_ford_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  struct Edge {
    int src, dest, weight;
  };

  std::vector<Edge> edges_;
  std::vector<int> distances_;
  int vertices_{0}, edges_count_{0}, source_{0};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  struct Edge {
    int src, dest, weight;

    template <class Archive>
    void serialize(Archive& ar, unsigned version) {
      ar & src & dest & weight;
    }
  };

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Edge> edges_;
  std::vector<int> distances_;
  int vertices_{0}, edges_count_{0}, source_{0};
  boost::mpi::communicator world;
};
}  // namespace vavilov_v_bellman_ford_mpi
