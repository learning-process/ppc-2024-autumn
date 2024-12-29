// Filateva Elizaveta Radix Sort

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace filateva_e_radix_sort_mpi {

class RadixSort : public ppc::core::Task {
 public:
  explicit RadixSort(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  int size;
  std::vector<int> arr;
  std::vector<int> ans;
};

}  // namespace filateva_e_radix_sort_mpi