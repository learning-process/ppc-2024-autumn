#pragma once

#include <gtest/gtest.h>
#include <random>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_dejskras_a_mpi {

std::vector<int> getRandomgraph(int v);
int minDistanceVertex(const std::vector<int>& dist, const std::vector<int>& marker);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static void dijkstra(int start_vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                       const std::vector<int>& data, int v, std::vector<int>& distances);
  static void get_children_with_weights(int vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                                        const std::vector<int>& data, std::vector<int>& neighbor,
                                        std::vector<int>& weight);

 private:
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> data;
  int v{};  // dimension
  std::vector<int> distances;
};
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> data;
  int v{};  // dimension
  std::vector<int> distances;

  boost::mpi::communicator world;
};

}  // namespace laganina_e_dejskras_a_mpi