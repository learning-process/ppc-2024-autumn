
#include "seq/Odintsov_M_VerticalRibbon_seq/include/ops_seq.hpp"

using namespace std::chrono_literals;

bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::validation() {
  internal_order_test();
  // if data is empty
  if ((taskData->inputs_count[0] == 0) || (taskData->inputs_count[2] == 0) || (taskData->outputs_count[0] == 0))
    return false;

  // if rowB != colA
  if ((taskData->inputs_count[0] / taskData->inputs_count[1]) != (taskData->inputs_count[2])) return false;

  // matrixA can be matrix
  if ((taskData->inputs_count[0] % taskData->inputs_count[1]) != 0) return false;
  return true;
}

bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::pre_processing() {
  internal_order_test();
  // [0] - szA [1] - count row A [2] - szB
  szA = taskData->inputs_count[0];
  rowA = taskData->inputs_count[1];
  colA = szA / rowA;
  szB = taskData->inputs_count[2];

  matrixA.assign(reinterpret_cast<double*>(taskData->inputs[0]), reinterpret_cast<double*>(taskData->inputs[0]) + szA);
  vectorB.assign(reinterpret_cast<double*>(taskData->inputs[1]), reinterpret_cast<double*>(taskData->inputs[1]) + szB);

  vectorC.assign(rowA, 0);
  return true;
}
bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::run() {
  internal_order_test();
  std::vector<double> ribbon(rowA, 0);

  // for ribbon
  for (int i = 0; i < colA; i++) {
    // create ribbon
    for (int j = 0; j < rowA; j++) {
      ribbon[j] = matrixA[colA * j + i];
    }
    // calculate
    for (int k = 0; k < rowA; k++) {
      vectorC[k] += ribbon[k] * vectorB[i];
    }
  }
  return true;
}
bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::post_processing() {
  internal_order_test();
  int szC = vectorC.size();
  for (int i = 0; i < szC; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = vectorC[i];
  }
  return true;
}
