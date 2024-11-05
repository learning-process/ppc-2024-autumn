#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_nearest_neighbor_elements_mpi {

struct LocalResult {
  int min_diff;
  int index1;
  int index2;

  bool operator<(const LocalResult& other) const {
    if (min_diff != other.min_diff) {
      return min_diff < other.min_diff;
    }
    return index1 < other.index1;
  }

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & min_diff;
    ar & index1;
    ar & index2;
  }
};

std::vector<int> getRandomVector(int sz);
std::pair<std::vector<int>, std::vector<int>> partitionArray(int amount, int num_partitions);

class FindClosestNeighborsSequentialMPI : public ppc::core::Task {
 public:
  explicit FindClosestNeighborsSequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int min_diff_{};
  int index1_{};
  int index2_{};
};

class FindClosestNeighborsParallelMPI : public ppc::core::Task {
 public:
  explicit FindClosestNeighborsParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int rank_offset_;
  int min_diff_ = std::numeric_limits<int>::max();
  int index1_ = -1;
  int index2_ = -1;
  std::vector<int> distribution;
  std::vector<int> displacement;
  boost::mpi::communicator world;
};

}  // namespace vasilev_s_nearest_neighbor_elements_mpi
