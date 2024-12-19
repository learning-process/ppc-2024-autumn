#include <mpi.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi {

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
  void setInputData(const std::vector<double>& input);
  const std::vector<double>& getSortedData() const;
  static std::vector<double> generateRandomData(int size, double minValue = -10.0, double maxValue = 10.0);

 private:
  std::vector<double> input_data_;
  std::vector<double> sorted_data_;
  void parallelSort();

  int rank;
  int size;
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void setInputData(const std::vector<double>& input);
  const std::vector<double>& getSortedData() const;

 private:
  std::vector<double> input_data_;
  std::vector<double> sorted_data_;
  void sequentialSort();
};

void radixSortWithSignHandling(std::vector<double>& data);
void radixSort(std::vector<double>& data, int num_bits, int radix);

}  // namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi
