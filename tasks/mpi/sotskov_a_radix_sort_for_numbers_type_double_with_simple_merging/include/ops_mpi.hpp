#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_);
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
  void radixSortWithSignHandling(std::vector<double>& data);
  static void radixSort(std::vector<double>& data, int num_bits, int radix);
  void parallelSort();
  boost::mpi::communicator world;
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_);
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void setInputData(const std::vector<double>& input);
  const std::vector<double>& getSortedData() const;

 private:
  std::vector<double> input_data_;
  std::vector<double> sorted_data_;
  void radixSortWithSignHandling(std::vector<double>& data);
  static void radixSort(std::vector<double>& data, int num_bits, int radix);
  void sequentialSort();
};

}  // namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi
