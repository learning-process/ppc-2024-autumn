#include "mpi/lopatin_i_quick_batcher_mergesort/include/quickBatcherMergesortHeaderMPI.hpp"

namespace lopatin_i_quick_batcher_mergesort_mpi {

void quicksort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pivotIndex = partition(arr, low, high);

    quicksort(arr, low, pivotIndex - 1);
    quicksort(arr, pivotIndex + 1, high);
  }
}

int partition(std::vector<int>& arr, int low, int high) {
  int pivot = arr[high];
  int i = low - 1;

  for (int j = low; j < high; j++) {
    if (arr[j] < pivot) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  return i + 1;
}

void oddEvenMerge(std::vector<int>& arr, int low, int high) {
  if (high <= low) return;

  int mid = (low + high) / 2;
  oddEvenMerge(arr, low, mid);
  oddEvenMerge(arr, mid + 1, high);

  // Merge the two halves
  for (int i = low; i <= mid; i++) {
    if (arr[i] > arr[i + (mid - low + 1)]) {
      std::swap(arr[i], arr[i + (mid - low + 1)]);
    }
  }
}

bool TestMPITaskSequential::validation() {
  internal_order_test();

  sizeArray = taskData->inputs_count[0];
  int sizeResultArray = taskData->outputs_count[0];

  if (sizeArray < 2 || sizeArray != sizeResultArray) {
    return false;
  }

  return true;
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();

  inputArray_.resize(sizeArray);
  resultArray_.resize(sizeArray);

  int* inputData = reinterpret_cast<int*>(taskData->inputs[0]);

  inputArray_.assign(inputData, inputData + sizeArray);

  return true;
}

bool TestMPITaskSequential::run() {
  internal_order_test();

  resultArray_.assign(inputArray_.begin(), inputArray_.end());
  quicksort(resultArray_, 0, sizeArray - 1);

  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();

  int* outputData = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(resultArray_.begin(), resultArray_.end(), outputData);

  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    sizeArray = taskData->inputs_count[0];
    int sizeResultArray = taskData->outputs_count[0];

    if (sizeArray < 2 || sizeArray != sizeResultArray) {
      return false;
    }
  }

  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    inputArray_.resize(sizeArray);
    resultArray_.resize(sizeArray);

    int* inputData = reinterpret_cast<int*>(taskData->inputs[0]);

    inputArray_.assign(inputData, inputData + sizeArray);
  }

  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* outputData = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(resultArray_.begin(), resultArray_.end(), outputData);
  }

  return true;
}

}  // namespace lopatin_i_quick_batcher_mergesort_mpi