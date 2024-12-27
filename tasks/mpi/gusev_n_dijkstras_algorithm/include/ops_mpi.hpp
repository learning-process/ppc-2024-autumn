#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gusev_n_dijkstras_algorithm_mpi {
class DijkstrasAlgorithmParallel : public ppc::core::Task {
 public:
  explicit DijkstrasAlgorithmParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  struct SparseGraphCRS {
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int num_vertices;

    SparseGraphCRS(int n) : num_vertices(n) { row_ptr.resize(n + 1, 0); }

    void add_edge(int u, int v, double weight) {
      values.push_back(weight);
      col_indices.push_back(v);

      for (size_t i = u + 1; i < row_ptr.size(); ++i) {
        row_ptr[i]++;
      }
    }
  };
  struct MinVertex {
    double distance;
    int vertex;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar & distance;
      ar & vertex;
    }

    bool operator<(const MinVertex& other) const { return distance < other.distance; }
  };

 private:
  boost::mpi::communicator world;
  std::vector<double> local_distances;
};

}  // namespace gusev_n_dijkstras_algorithm_mpi
