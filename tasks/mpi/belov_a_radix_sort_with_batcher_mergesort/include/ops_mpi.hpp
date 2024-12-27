#ifndef OPS_MPI_HPP
#define OPS_MPI_HPP

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

using bigint = long long;
using namespace std;

namespace belov_a_radix_batcher_mergesort_mpi {

class RadixBatcherMergesortParallel : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static void sort(vector<bigint>& arr);

 private:
  boost::mpi::communicator world;
  vector<bigint> array;  // input unsorted numbers array
  int n = 0;             // array size

  static vector<bigint> merge(const vector<bigint>& arr1, const vector<bigint>& arr2);
  static void compare_and_swap(vector<int>& data, int i, int j);
  static vector<bigint> odd_even_merge(const vector<bigint>& left, const vector<bigint>& right);

  static void radix_sort(vector<bigint>& arr, bool invert);
  static void counting_sort(vector<bigint>& arr, bigint digit_place);
  static int get_number_digit_capacity(bigint num);
};

class RadixBatcherMergesortSequential : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static void sort(vector<bigint>& arr);

 private:
  vector<bigint> array;
  int n = 0;

  static void radix_sort(vector<bigint>& arr, bool invert);
  static void counting_sort(vector<bigint>& arr, bigint digit_place);
  static int get_number_digit_capacity(bigint num);
};

}  // namespace belov_a_radix_batcher_mergesort_mpi

#endif  // OPS_MPI_HPP