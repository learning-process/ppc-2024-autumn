#include "seq/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderSeq.hpp"

namespace lopatin_i_strip_horizontal_scheme_seq {

bool TestTaskSequential::validation() {
  internal_order_test();

  sizeX = taskData->inputs_count[0];
  sizeY = taskData->inputs_count[1];
  int vectorSize = taskData->inputs_count[2];

  return (sizeX > 0 && sizeY > 0 && vectorSize > 0 && sizeX == vectorSize);
}

bool TestTaskSequential::pre_processing() {
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

bool TestTaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < sizeY; i++) {
    for (int j = 0; j < sizeX; j++) {
      resultVector_[i] += matrix_[i * sizeX + j] * vector_[j];
    }
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  int* outputData = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(resultVector_.begin(), resultVector_.end(), outputData);

  return true;
}

}  // namespace lopatin_i_strip_horizontal_scheme_seq