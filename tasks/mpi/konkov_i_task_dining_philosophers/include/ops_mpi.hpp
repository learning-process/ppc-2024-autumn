#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_task_dining_philosophers {

class DiningPhilosophersMPITaskParallel : public ppc::core::Task {
 public:
  explicit DiningPhilosophersMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<int> input_, local_input_;
  int res{};
  std::vector<int> getRandomVector(int sz);
};

}  // namespace konkov_i_task_dining_philosophers
