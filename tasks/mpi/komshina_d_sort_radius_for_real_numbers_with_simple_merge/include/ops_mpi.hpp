#include <mpi.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  };
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input;
  std::vector<double> sort;
  
  void ParallelSortDouble();
  int rank;
  int size;
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)){};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input;
  std::vector<double> sort;
};
void SortDouble(std::vector<double>& data);
void CountingSort(double* inp, double* out, int byteNum, int size);

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi