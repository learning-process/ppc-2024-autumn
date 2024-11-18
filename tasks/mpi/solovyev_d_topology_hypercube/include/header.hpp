#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovyev_d_topology_hypercube_mpi {

std::vector<int> intToBin(int number, int padding);
int binToInt(std::vector<int> binary);
int calcAddress(int src, int dest, int maxAddressLength);

class TopologyHypercubeMPI : public ppc::core::Task {
 public:
  explicit TopologyHypercubeMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  struct data {
    int value;
    bool finish;
    int destination;
    std::vector<int> path;
    template <class Archive>
    void serialize(Archive& ar, unsigned int version) {
      ar& value;
      ar& finish;
      ar& destination;
      ar& path;
    }
  } data_;

  int maxAddressLength{};
  int value{};
  int destination{};

  std::vector<int> input_;
  boost::mpi::communicator world;
};

}  // namespace solovyev_d_topology_hypercube_mpi