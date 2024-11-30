#include "seq/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderSeq.hpp"

namespace lopatin_i_strip_horizontal_scheme_seq {

std::vector<int> generateVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> outputVector(size);
  for (int i = 0; i < size; i++) {
    outputVector[i] = (gen() % 200);
  }
  return outputVector;
}

std::vector<int> generateMatrix(int sizeX, int sizeY) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int matrixSize = sizeX * sizeY;
  std::vector<int> outputMatrix(matrixSize);
  for (int i = 0; i < matrixSize; i++) {
    outputMatrix[i] = (gen() % 200);
  }
  return outputMatrix;
}

bool lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential::validation() {
  internal_order_test();

  sizeX = taskData->inputs_count[0];
  sizeY = taskData->inputs_count[1];
  int vectorSize = taskData->inputs_count[2];

  return (sizeX > 0 && sizeY > 0 && vectorSize > 0 && sizeX == vectorSize);
}

bool lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential::pre_processing() { 
  internal_order_test();

  matrix_.resize(sizeX * sizeY);
  vector_.resize(sizeX);
  resultVector_.resize(sizeY);

  int* matrixData = reinterpret_cast<int*>(taskData->inputs[0]);
  int* vectorData = reinterpret_cast<int*>(taskData->inputs[1]);

  matrix_.assign(matrixData, matrixData + sizeX * sizeY);
  vector_.assign(vectorData, vectorData + sizeX);

  return true;
}

bool lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < sizeY; i++) {
    for (int j = 0; j < sizeX; j++) {
      resultVector_[i] += matrix_[i * sizeX + j] * vector_[j];
    }
  }

  return true;
}

bool lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* outputData = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(resultVector_.begin(), resultVector_.end(), outputData);

  return true;
}

}  // namespace lopatin_i_strip_horizontal_scheme_seq