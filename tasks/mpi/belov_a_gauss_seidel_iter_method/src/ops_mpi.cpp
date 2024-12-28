#include "mpi/belov_a_gauss_seidel_iter_method/include/ops_mpi.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "boost/mpi/detail/broadcast_sc.hpp"

using namespace std;

namespace belov_a_gauss_seidel_mpi {

bool GaussSeidelParallel::ge_double(double a, double b) {
  return (a - b) >= -((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * std::numeric_limits<double>::epsilon());
}

bool GaussSeidelParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    n = taskData->inputs_count[0];

    auto* inputMatrixData = reinterpret_cast<double*>(taskData->inputs[0]);
    A.assign(inputMatrixData, inputMatrixData + n * n);

    auto* freeMembersVector = reinterpret_cast<double*>(taskData->inputs[1]);
    b.assign(freeMembersVector, freeMembersVector + n);

    epsilon = *(reinterpret_cast<double*>(taskData->inputs[2]));
  }
  x.assign(n, 0.0);  // initial approximations

  return true;
}

bool GaussSeidelParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.size() != 3 || taskData->inputs_count.empty()) return false;

    vector<double> mt;
    auto* mt_data = reinterpret_cast<double*>(taskData->inputs[0]);
    mt.assign(mt_data, mt_data + taskData->inputs_count[0] * taskData->inputs_count[0]);

    return (!taskData->outputs.empty() && (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
            taskData->inputs_count[0] * taskData->inputs_count[0] == taskData->inputs_count[2] &&
            isDiagonallyDominant(mt, taskData->inputs_count[0]));
  }
  return true;
}

vector<int> caclulate_sizes(size_t input_size, size_t n) {
  size_t count = input_size / n;
  size_t mod = input_size % n;
  vector<int> sizes(n, count);

  transform(sizes.cbegin(), sizes.cbegin() + mod, sizes.begin(), [](auto i) { return i + 1; });

  return sizes;
}

vector<int> caclulate_displ(size_t input_size, size_t n) {
  auto sizes = caclulate_sizes(input_size, n);
  vector<int> displ(n, 0);

  for (size_t i = 1; i < n; ++i) {
    displ[i] = displ[i - 1] + sizes[i - 1];
  }

  return displ;
}

vector<vector<double>> calculate_rows_per_process(const vector<double>& matrix, int world_size) {
  vector<vector<double>> result(world_size);
  int n = sqrt(matrix.size());
  auto sizes = caclulate_sizes(n, world_size);
  auto displacements = caclulate_displ(n, world_size);

  for (int k = 0; k < world_size - 1; ++k) {
    for (int i = 0; i < n; ++i) {
      for (int j = displacements[k]; j < displacements[k + 1]; ++j) {
        result[k].push_back(matrix[i * n + j]);
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = displacements.back(); j < n; ++j) {
      result[world_size - 1].push_back(matrix[i * n + j]);
    }
  }

  return result;
}

bool GaussSeidelParallel::run() {
  internal_order_test();

  vector<double> local_data;

  int rank = world.rank();
  int local_size;
  int local_displ;

  boost::mpi::broadcast(world, x, 0);
  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, epsilon, 0);

  if (world.rank() == 0) {
    auto rows = calculate_rows_per_process(A, world.size());
    auto sizes = caclulate_sizes(n, world.size());
    auto displ = caclulate_displ(n, world.size());

    local_data = rows[0];
    local_size = sizes[0];
    local_displ = displ[0];

    for (int i = 1; i < world.size(); ++i) {
      world.send(i, 0, rows[i]);
      world.send(i, 1, sizes[i]);
      world.send(i, 1, displ[i]);
    }
  } else {
    world.recv(0, 0, local_data);
    world.recv(0, 1, local_size);
    world.recv(0, 1, local_displ);
  }

  vector<double> x_new(n, 0.0);

  do {
    norm = 0.0;
    int d = 0;

    for (int i = 0; i < n; ++i) {
      double loc_sum = 0.0;

      for (int j = d, X = local_displ; j < d + local_size; ++j, ++X) {
        if (i != X) {
          loc_sum += local_data[j] * x_new[X];
        }
      }

      d += local_size;
      double sum;

      boost::mpi::reduce(world, loc_sum, sum, plus(), 0);

      if (rank == 0) {
        x_new[i] = (b[i] - sum) / A[i * n + i];
      }

      boost::mpi::broadcast(world, x_new[i], 0);
    }

    for (int i = 0; i < n; ++i) {
      norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
    }

    norm = sqrt(norm);
    x = x_new;

  } while (ge_double(norm, epsilon));

  return true;
}

bool GaussSeidelParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    copy(x.begin(), x.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }

  return true;
}

bool isDiagonallyDominant(const vector<double>& A, int n) {
  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) row_sum += abs(A[i * n + j]);
    }

    if (abs(A[i * n + i]) <= row_sum) {
      return false;
    }
  }

  return true;
}

bool GaussSeidelSequential::ge_double(double a, double b) {
  return (a - b) >= -((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * std::numeric_limits<double>::epsilon());
}

bool GaussSeidelSequential::pre_processing() {
  internal_order_test();

  n = taskData->inputs_count[0];
  auto* inputMatrixData = reinterpret_cast<double*>(taskData->inputs[0]);
  A.assign(inputMatrixData, inputMatrixData + n * n);

  auto* freeMembersVector = reinterpret_cast<double*>(taskData->inputs[1]);
  b.assign(freeMembersVector, freeMembersVector + n);

  epsilon = *(reinterpret_cast<double*>(taskData->inputs[2]));

  x.assign(n, 0.0);  // initial approximations

  return true;
}

bool GaussSeidelSequential::validation() {
  internal_order_test();

  if (taskData->inputs.size() != 3 || taskData->inputs_count.size() < 3 ||
      taskData->inputs_count[0] * taskData->inputs_count[0] != taskData->inputs_count[2])
    return false;

  vector<double> mt;
  auto* mt_data = reinterpret_cast<double*>(taskData->inputs[0]);
  mt.assign(mt_data, mt_data + taskData->inputs_count[0] * taskData->inputs_count[0]);

  return (!taskData->outputs.empty() && (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
          isDiagonallyDominant(mt, taskData->inputs_count[0]));
}

bool GaussSeidelSequential::run() {
  internal_order_test();

  vector<double> x_new(n, 0.0);
  double norm;

  do {
    norm = 0.0;

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i * n + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i * n + i];
    }

    for (int i = 0; i < n; ++i) {
      norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
      x[i] = x_new[i];
    }
    norm = sqrt(norm);

  } while (ge_double(norm, epsilon));

  return true;
}

bool GaussSeidelSequential::post_processing() {
  internal_order_test();

  copy(x.begin(), x.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

}  // namespace belov_a_gauss_seidel_mpi