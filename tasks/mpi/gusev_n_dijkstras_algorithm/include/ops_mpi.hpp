#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
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

      for (int i = u + 1; i < row_ptr.size(); ++i) {
        row_ptr[i]++;
      }
    }

    void print_graph() const {
      std::cout << "Values: ";
      for (const auto& value : values) {
        std::cout << value << " ";
      }
      std::cout << "\nColumn Indices: ";
      for (const auto& index : col_indices) {
        std::cout << index << " ";
      }
      std::cout << "\nRow Pointers: ";
      for (const auto& ptr : row_ptr) {
        std::cout << ptr << " ";
      }
      std::cout << std::endl;
    }

    void print_graph_ascii() const {
      std::cout << "Graph:\n";
      for (int i = 0; i < num_vertices; ++i) {
        std::cout << "Vershina " << i << ": ";
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
          std::cout << " -> " << col_indices[j] << " (Ves: " << values[j] << ")";
        }
        std::cout << std::endl;
      }
    }
  };

 private:
  boost::mpi::communicator world;
};

}  // namespace gusev_n_dijkstras_algorithm_mpi
