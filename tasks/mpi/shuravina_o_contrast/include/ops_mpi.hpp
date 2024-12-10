#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_contrast {

class ContrastTaskParallel : public ppc::core::Task {
 public:
  explicit ContrastTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_, local_input_, output_;
  uint8_t min_val_, max_val_;
  boost::mpi::communicator world;
};

}  // namespace shuravina_o_contrast