#pragma once
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>
#include <iterator>
#include <algorithm> 

#include "core/task/include/task.hpp"
namespace kudryashova_i_gather_my {
int vectorDotProductGather(const std::vector<int>& vector1, const std::vector<int>& vector2);
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
 private:
  std::vector<int> input_data;
  std::vector<int> firstHalf, secondHalf;
  int reference{};
};
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_): Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  template <typename T>
  void gather_my(const boost::mpi::communicator& wrld, const T& local_data, std::vector<T>& full_result,
                        int root);
 private:
  boost::mpi::communicator world;
  std::vector<int> input_data;
  std::vector<int> local_input1_, local_input2_;
  std::vector<int> firstHalf, secondHalf;

  int result{};
  int local_result;
  unsigned int delta{};
};
}  // namespace kudryashova_i_gather_my