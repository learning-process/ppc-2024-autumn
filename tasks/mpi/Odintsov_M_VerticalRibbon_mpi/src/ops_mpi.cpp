#include "mpi/Odintsov_M_VerticalRibbon_mpi/include/ops_mpi.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
using namespace std::chrono_literals;

namespace Odintsov_M_VerticalRibbon_mpi {
// Последовательная версия
bool VerticalRibbonMPISequential::validation() {
  internal_order_test();
  if ((taskData->inputs_count[0] == 0) || (taskData->inputs_count[2] == 0) || (taskData->outputs_count[0] == 0))
    return false;

  // if rowB != colA
  if ((taskData->inputs_count[0] / taskData->inputs_count[1]) != (taskData->inputs_count[2])) return false;

  // matrixA can be matrix
  if ((taskData->inputs_count[0] % taskData->inputs_count[1]) != 0) return false;
  return true;
}
bool VerticalRibbonMPISequential::pre_processing() {
  internal_order_test();
  // [0] - szA [1] - count row A [2] - szB
  szA = taskData->inputs_count[0];
  rowA = taskData->inputs_count[1];
  colA = szA / rowA;
  szB = taskData->inputs_count[2];

  matrixA.assign(reinterpret_cast<double *>(taskData->inputs[0]),
                 reinterpret_cast<double *>(taskData->inputs[0]) + szA);
  vectorB.assign(reinterpret_cast<double *>(taskData->inputs[1]),
                 reinterpret_cast<double *>(taskData->inputs[1]) + szB);

  vectorC.assign(rowA, 0);
  return true;
}
bool VerticalRibbonMPISequential::run() {
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
bool VerticalRibbonMPISequential::post_processing() {
  internal_order_test();
  int szC = vectorC.size();
  for (int i = 0; i < szC; i++) {
    reinterpret_cast<double *>(taskData->outputs[0])[i] = vectorC[i];
  }
  return true;
}
// Parallel version
bool VerticalRibbonMPIParallel::validation() {
  internal_order_test();
  if (com.rank() == 0) {
    if ((taskData->inputs_count[0] == 0) || (taskData->inputs_count[2] == 0) || (taskData->outputs_count[0] == 0))
      return false;
    // if rowB != colA
    if ((taskData->inputs_count[0] / taskData->inputs_count[1]) != (taskData->inputs_count[2])) return false;
    // matrixA can be matrix
    if ((taskData->inputs_count[0] % taskData->inputs_count[1]) != 0) return false;
  }

  return true;
}

bool VerticalRibbonMPIParallel::pre_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    // [0] - szA [1] - count row A [2] - szB
    szA = taskData->inputs_count[0];
    rowA = taskData->inputs_count[1];
    colA = szA / rowA;
    szB = taskData->inputs_count[2];

    matrixA.assign(reinterpret_cast<double *>(taskData->inputs[0]),
                   reinterpret_cast<double *>(taskData->inputs[0]) + szA);
    vectorB.assign(reinterpret_cast<double *>(taskData->inputs[1]),
                   reinterpret_cast<double *>(taskData->inputs[1]) + szB);

    vectorC.assign(rowA, 0);
  }

  return true;
}
bool VerticalRibbonMPIParallel::run() {
  internal_order_test();

  if (com.rank() == 0) {
    ribbon_sz = std::max(1, (colA + com.size() - 1) / com.size());
  }

  // fflush(stdout);
  broadcast(com, ribbon_sz, 0);
  broadcast(com, rowA, 0);
  broadcast(com, colA, 0);
  vectorB.resize(colA, 0);

  broadcast(com, vectorB.data(), colA, 0);

  if (com.rank() == 0) {
    for (int pr = 1; pr < com.size(); pr++) {
      std::vector<double> ribbon;
      int startcol = pr * ribbon_sz;
      int endcol = (pr + 1) * ribbon_sz;
      if (endcol <= colA) {
        for (int j = 0; j < rowA; j++) {
          for (int i = startcol; i < endcol; i++) {
            ribbon.push_back(matrixA[colA * j + i]);
          }
        }
        com.send(pr, 0, ribbon.data(), ribbon.size());
      }
    }
  }

  if (com.rank() == 0) {
    for (int j = 0; j < rowA; j++) {
      for (int i = 0; i < ribbon_sz; i++) {
        local_ribbon.push_back(matrixA[colA * j + i]);
      }
    }
  } else {
    if (com.rank() * ribbon_sz <= colA) {
      int endcol = (com.rank() + 1) * ribbon_sz;
      std::vector<double> buffer(ribbon_sz * rowA, 0);
      if (endcol <= colA) {
        com.recv(0, 0, buffer.data(), buffer.size());
        local_ribbon.insert(local_ribbon.end(), buffer.begin(), buffer.end());
      }
    }
  }

  // Calculate
  localC.assign(rowA, 0);
  if (!local_ribbon.empty()) {
    for (int i = 0; i < ribbon_sz; i++) {
      for (int j = 0; j < rowA; j++) {
        localC[j] += local_ribbon[ribbon_sz * j + i] * vectorB[com.rank() * ribbon_sz + i];
      }
    }
  }
  size_t szC = localC.size();
  for (size_t k = 0; k < szC; k++) {
    reduce(com, localC[k], vectorC[k], std::plus(), 0.0);
  }

  return true;
}

bool VerticalRibbonMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    size_t szC = vectorC.size();
    for (size_t i = 0; i < szC; i++) {
      reinterpret_cast<double *>(taskData->outputs[0])[i] = vectorC[i];
    }
  }
  return true;
}
};  // namespace Odintsov_M_VerticalRibbon_mpi
