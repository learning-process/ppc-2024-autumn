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

bool TestMPITaskSequential::validation() {
  internal_order_test();

  sizeArray = taskData->inputs_count[0];
  int sizeResultArray = taskData->outputs_count[0];

  return (sizeArray > 1 && sizeArray == sizeResultArray);
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

    return (sizeArray > 1 && sizeArray == sizeResultArray);
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

  boost::mpi::broadcast(world, sizeArray, 0);

  int chunkSize = sizeArray / world.size();
  int remainder = sizeArray % world.size();

  int startPosition = world.rank() * chunkSize;
  int actualChunkSize = chunkSize;

  if (world.rank() < remainder) {
    startPosition += world.rank();
    actualChunkSize++;
  } else {
    startPosition += remainder;
  }

  localArray.resize(actualChunkSize);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      int procStartPosition = proc * chunkSize;
      int procActualChunkSize = chunkSize;
      if (proc < remainder) {
        procStartPosition += proc;
        procActualChunkSize++;
      } else {
        procStartPosition += remainder;
      }
      if (procActualChunkSize > 0) {
        world.send(proc, 0, inputArray_.data() + procStartPosition, procActualChunkSize);
      }
    }
    std::copy(inputArray_.begin() + startPosition, inputArray_.begin() + startPosition + actualChunkSize,
              localArray.begin());
  } else {
    if (actualChunkSize > 0) {
      world.recv(0, 0, localArray.data(), actualChunkSize);
    }
  }

  quicksort(localArray, 0, localArray.size() - 1);

  for (int oddEvenStep = 0; oddEvenStep < world.size(); oddEvenStep++) {
    int border = world.size();
    if (oddEvenStep % 2 == 0) {  // odd
      if (world.size() % 2 != 0) {
        border -= 1;
      }
      if (world.rank() >= border) {
        continue;
      }

      std::vector<int> recvArray(actualChunkSize);

      if (world.rank() % 2 == 0) {
        world.send(world.rank() + 1, 0, localArray);
        world.recv(world.rank() + 1, 1, recvArray);
        std::vector<int> mergedArrays(localArray.size() + recvArray.size());
        std::merge(localArray.begin(), localArray.end(), recvArray.begin(), recvArray.end(), mergedArrays.begin());
        localArray = mergedArrays;
        localArray.resize(actualChunkSize);
      } else {
        world.recv(world.rank() - 1, 0, recvArray);
        world.send(world.rank() - 1, 1, localArray);
        std::vector<int> mergedArrays(localArray.size() + recvArray.size());
        std::merge(localArray.begin(), localArray.end(), recvArray.begin(), recvArray.end(), mergedArrays.begin());
        localArray = mergedArrays;
        localArray.erase(localArray.begin(), localArray.begin() + localArray.size() / 2);
      }
    } else {  // even
      if (world.size() % 2 == 0) {
        border -= 1;
      }
      if (world.rank() < 1 || world.rank() >= border) {
        continue;
      }

      std::vector<int> recvArray(actualChunkSize);

      if (world.rank() % 2 != 0) {
        world.send(world.rank() + 1, 0, localArray);
        world.recv(world.rank() + 1, 1, recvArray);
        std::vector<int> mergedArrays(localArray.size() + recvArray.size());
        std::merge(localArray.begin(), localArray.end(), recvArray.begin(), recvArray.end(), mergedArrays.begin());
        localArray = mergedArrays;
        localArray.resize(actualChunkSize);
      } else {
        world.recv(world.rank() - 1, 0, recvArray);
        world.send(world.rank() - 1, 1, localArray);
        std::vector<int> mergedArrays(localArray.size() + recvArray.size());
        std::merge(localArray.begin(), localArray.end(), recvArray.begin(), recvArray.end(), mergedArrays.begin());
        localArray = mergedArrays;
        localArray.erase(localArray.begin(), localArray.begin() + localArray.size() / 2);
      }
    }
  }

  if (world.rank() == 0) {
    boost::mpi::gather(world, localArray.data(), actualChunkSize, resultArray_, 0);
  } else {
    boost::mpi::gather(world, localArray.data(), actualChunkSize, 0);
  }

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