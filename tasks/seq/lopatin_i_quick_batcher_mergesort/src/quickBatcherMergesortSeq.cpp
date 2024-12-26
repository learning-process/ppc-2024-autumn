#include "seq/lopatin_i_quick_batcher_mergesort/include/quickBatcherMergesortHeaderSeq.hpp"

namespace lopatin_i_quick_batcher_mergesort_seq {

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

bool TestTaskSequential::validation() {
  internal_order_test();

  sizeArray = taskData->inputs_count[0];
  int sizeResultArray = taskData->outputs_count[0];

  return (sizeArray > 1 && sizeArray == sizeResultArray);

  return true;
}

bool TestTaskSequential::pre_processing() {
  internal_order_test();

  inputArray_.resize(sizeArray);
  resultArray_.resize(sizeArray);

  int* inputData = reinterpret_cast<int*>(taskData->inputs[0]);

  inputArray_.assign(inputData, inputData + sizeArray);

  return true;
}

bool TestTaskSequential::run() {
  internal_order_test();

  resultArray_.assign(inputArray_.begin(), inputArray_.end());

  quicksort(resultArray_, 0, sizeArray - 1);

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  int* outputData = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(resultArray_.begin(), resultArray_.end(), outputData);

  return true;
}

}  // namespace lopatin_i_quick_batcher_mergesort_seq