#include "mpi/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderMPI.hpp"

namespace lopatin_i_strip_horizontal_scheme_mpi {

std::vector<int> generateVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> outputVector(size);
  for (int i = 0; i < size; i++) {
    outputVector[i] = (gen() % 200) - 99;
  }
  return outputVector;
}

std::vector<int> generateMatrix(int sizeX, int sizeY) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int matrixSize = sizeX * sizeY;
  std::vector<int> outputMatrix(matrixSize);
  for (int i = 0; i < matrixSize; i++) {
    outputMatrix[i] = (gen() % 200) - 99;
  }
  return outputMatrix;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();

  sizeX = taskData->inputs_count[0];
  sizeY = taskData->inputs_count[1];
  int vectorSize = taskData->inputs_count[2];

  return (sizeX > 0 && sizeY > 0 && vectorSize > 0 && sizeX == vectorSize);
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();

  matrix_.resize(sizeX * sizeY);
  vector_.resize(sizeX);
  resultVector_.resize(sizeY);

  int *matrixData = reinterpret_cast<int *>(taskData->inputs[0]);
  int *vectorData = reinterpret_cast<int *>(taskData->inputs[1]);

  matrix_.assign(matrixData, matrixData + sizeX * sizeY);
  vector_.assign(vectorData, vectorData + sizeX);

  return true;
}

bool TestMPITaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < sizeY; i++) {
    for (int j = 0; j < sizeX; j++) {
      resultVector_[i] += matrix_[i * sizeX + j] * vector_[j];
    }
  }

  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();

  int *outputData = reinterpret_cast<int *>(taskData->outputs[0]);
  std::copy(resultVector_.begin(), resultVector_.end(), outputData);

  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    sizeX = taskData->inputs_count[0];
    sizeY = taskData->inputs_count[1];
    int vectorSize = taskData->inputs_count[2];

    return (sizeX > 0 && sizeY > 0 && vectorSize > 0 && sizeX == vectorSize);
  }

  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    matrix_.resize(sizeX * sizeY);
    vector_.resize(sizeX);
    resultVector_.resize(sizeY);

    int *matrixData = reinterpret_cast<int *>(taskData->inputs[0]);
    int *vectorData = reinterpret_cast<int *>(taskData->inputs[1]);

    matrix_.assign(matrixData, matrixData + sizeX * sizeY);
    vector_.assign(vectorData, vectorData + sizeX);
  }

  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, sizeX, 0);
  boost::mpi::broadcast(world, sizeY, 0);

  int chunkSize = sizeY / world.size();
  int startRow = world.rank() * chunkSize;
  int actualChunkSize = (world.rank() == world.size() - 1) ? (sizeY - startRow) : chunkSize;

  localMatrix_.resize(sizeX * actualChunkSize);
  std::vector<int> localVector(sizeX);

  if (world.rank() == 0) {
    localVector = vector_;
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, localVector.data(), sizeX);
    }
  } else {
    world.recv(0, 0, localVector.data(), sizeX);
  }

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      int procStartRow = proc * chunkSize;
      int procActualChunkSize = (proc == world.size() - 1) ? (sizeY - procStartRow) : chunkSize;
      if (procActualChunkSize > 0) {
        world.send(proc, 0, matrix_.data() + procStartRow * sizeX, procActualChunkSize * sizeX);
      }
    }
    std::copy(matrix_.begin(), matrix_.begin() + actualChunkSize * sizeX, localMatrix_.begin());
  } else {
    world.recv(0, 0, localMatrix_.data(), actualChunkSize * sizeX);
  }

  std::vector<int> localResult(actualChunkSize, 0);
  for (int i = 0; i < actualChunkSize; i++) {
    for (int j = 0; j < sizeX; j++) {
      localResult[i] += localMatrix_[i * sizeX + j] * localVector[j];
    }
  }
  boost::mpi::gather(world, localResult.data(), actualChunkSize, resultVector_.data(), 0);

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int *outputData = reinterpret_cast<int *>(taskData->outputs[0]);
    std::copy(resultVector_.begin(), resultVector_.end(), outputData);
  }

  return true;
}

}  // namespace lopatin_i_strip_horizontal_scheme_mpi