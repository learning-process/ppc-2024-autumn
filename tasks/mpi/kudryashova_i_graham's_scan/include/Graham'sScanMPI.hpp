#pragma once
#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>

#include "core/task/include/task.hpp"
namespace kudryashova_i_graham_scan_mpi {
class TestMPITaskSequential : public ppc::core::Task {
 public:
  std::vector<int8_t> runGrahamScan(std::vector<int8_t>& input_data);
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int8_t> input_data;
  std::vector<int> firstHalf, secondHalf;
  std::vector<int8_t> result_vec;
};
class TestMPITaskParallel : public ppc::core::Task {
 public:
  std::vector<int8_t> runGrahamScan(std::vector<int8_t>& input_data);
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<int8_t> input_data;
  std::vector<int> firstHalf, secondHalf;
  std::vector<int> local_input1_, local_input2_;
  std::vector<int> segments;
  std::vector<int8_t> local_result;
  std::vector<int8_t> result_vec;
  int delta{};
  int processes{};
};
}  // namespace kudryashova_i_graham_scan_mpi
