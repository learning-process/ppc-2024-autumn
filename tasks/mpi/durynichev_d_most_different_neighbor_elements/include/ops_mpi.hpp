#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/serialization/serialization.hpp>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
namespace durynichev_d_most_different_neighbor_elements_mpi {

struct ChunkResult {
  size_t left_index;
  size_t right_index;
  int diff;

  template <class Archive>
  void serialize(Archive &archive, const unsigned int version) {
    archive & left_index;
    archive & right_index;
    archive & diff;
  }

  ChunkResult operator()(const ChunkResult &a, const ChunkResult &b) {
    return (a.diff > b.diff || (a.diff == b.diff && (a.left_index < b.left_index))) ? a : b;
  }

  std::vector<int> toVector(const std::vector<int> &input) const {
    return std::vector<int>{
        std::min(input[left_index], input[right_index]),
        std::max(input[left_index], input[right_index]),
    };
  }
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input, result;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<int> input, chunk;
  int chunkStart = 0;
  ChunkResult result{};
};

}  // namespace durynichev_d_most_different_neighbor_elements_mpi
