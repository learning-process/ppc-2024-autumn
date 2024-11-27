#pragma once
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_topology_hypercube_mpi {

int CalculateNextHop(int sourceRank, int targetRank, int maxAddressBits);
int BinaryToInt(std::vector<int> binary);
std::vector<int> IntToBinary(int number, int padding = 0);

class HypercubeRouterMPI : public ppc::core::Task {
 public:
  explicit HypercubeRouterMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  struct RoutingData {
    int payload;
    bool isFinished;
    int targetRank;
    std::vector<int> route;
    template <class Archive>
    void serialize(Archive& ar, unsigned int version) {
      ar & payload;
      ar & isFinished;
      ar & targetRank;
      ar & route;
    }
  };

 private:
  RoutingData routingData;
  int maxAddressBits{};
  boost::mpi::communicator world;
};

}  // namespace alputov_i_topology_hypercube_mpi