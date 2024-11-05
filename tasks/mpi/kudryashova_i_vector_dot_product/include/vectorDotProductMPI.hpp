#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>
#include <random>

#include "core/task/include/task.hpp"
namespace kudryashova_i_vector_dot_product_mpi {
int vectorDotProduct(const std::vector<int>& vector1, const std::vector<int>& vector2);
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  int result{};
};
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> local_input1_, local_input2_;
  int result{};
  boost::mpi::communicator world;
  unsigned int delta;
};
}  // namespace kudryashova_i_vector_dot_product_mpi
