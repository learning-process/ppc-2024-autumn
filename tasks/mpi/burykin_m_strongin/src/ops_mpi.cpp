#include "mpi/burykin_m_strongin/include/ops_mpi.hpp"

namespace burykin_m_strongin {

bool StronginSequential::pre_processing() {
  internal_order_test();
  res = 0;
  x0 = *reinterpret_cast<double*>(taskData->inputs[0]);
  x1 = *reinterpret_cast<double*>(taskData->inputs[1]);
  eps = *reinterpret_cast<double*>(taskData->inputs[2]);
  return true;
}

bool StronginSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool StronginSequential::run() {
  internal_order_test();
  std::vector<double> x;
  std::vector<double> y;
  double lipshM = 0.0;
  double lipshm = 0.0;
  double R = 0.0;
  size_t interval = 0;

  if (x1 > x0) {
    x.push_back(x0);
    x.push_back(x1);
  } else {
    x.push_back(x1);
    x.push_back(x0);
  }

  while (true) {
    for (size_t i = 0; i < x.size(); i++) {
      y.push_back(f(x[i]));
    }
    for (size_t i = 0; i < x.size() - 1; i++) {
      double lipsh = std::abs((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
      if (lipsh > lipshM) {
        lipshM = lipsh;
        lipshm = lipsh + lipsh;
        double tempR = lipshm * (x[i + 1] - x[i]) + pow((y[i + 1] - y[i]), 2) / (lipshm * (x[i + 1] - x[i])) -
                       2 * (y[i + 1] + y[i]);
        if (tempR > R) {
          R = tempR;
          interval = i;
        }
      }
    }
    if (x[interval + 1] - x[interval] <= eps) {
      res = y[interval + 1];
      return true;
    }
    double newX;
    newX = (x[interval + 1] - x[interval]) / 2 + x[interval] + (y[interval + 1] - y[interval]) / (2 * lipshm);
    x.push_back(newX);
    sort(x.begin(), x.end());
    y.clear();
  }
}

bool StronginSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool StronginParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    res = 0;
    x0 = *reinterpret_cast<double*>(taskData->inputs[0]);
    x1 = *reinterpret_cast<double*>(taskData->inputs[1]);
    eps = *reinterpret_cast<double*>(taskData->inputs[2]);
  }
  return true;
}

bool StronginParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool StronginParallel::run() {
  internal_order_test();

  int size;
  int rank;
  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double lipshM = 0;
  double lipshm = 0.0;
  double lipsh;

  int interval = 0;
  double R = 0.0;

  if (rank == 0) {
    std::vector<double> x;
    x.push_back(x0);
    x.push_back(x1);

    while (true) {
      sort(x.begin(), x.end());

      int part = static_cast<int>((x.size() - 1) / size);
      int remain = static_cast<int>((x.size() - 1) % size);

      if (part > 0) {
        for (int i = 1; i < size; ++i) {
          MPI_Send(x.data() + remain + (i - 1) * part, part, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
      }
      for (int i = 0; i < part + remain; ++i) {
        lipsh = std::abs((f(x[i + 1]) - f(x[i])) / (x[i + 1] - x[i]));
        if (lipsh > lipshM) {
          lipshM = lipsh;
          lipshm = 2 * lipsh;
        }
      }
      if (part > 0) {
        for (int i = 1; i < size; i++) {
          MPI_Recv(&lipsh, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          if (lipsh > lipshM) {
            lipshM = lipsh;
            lipshm = 2 * lipsh;
          }
        }
      }

      double tempR;
      for (int i = 0; i < static_cast<int>(x.size()) - 1; i++) {
        tempR = lipshm * (x[i + 1] - x[i]) + pow((f(x[i + 1]) - f(x[i])), 2) / (lipshm * (x[i + 1] - x[i])) -
                2 * (f(x[i + 1]) + f(x[i]));
        if (tempR > R) {
          R = tempR;
          interval = i;
        }
      }
      if (x[interval + 1] - x[interval] <= eps) {
        for (int i = 1; i < size; ++i) MPI_Send(x.data(), 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        res = x[interval + 1];
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
      }

      double newX = (x[interval] + x[interval + 1]) / 2 - (f(x[interval + 1]) - f(x[interval])) / (lipshm + lipshm);
      x.push_back(newX);
    }
  } else {
    int part = 0;
    while (true) {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &part);

      std::vector<double> x(part + 1);
      MPI_Recv(x.data(), part, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
      }

      // double lipshM = 0;
      // double lipshm = 1.0;
      // double lipsh;

      if (part != 0) {
        for (int i = 0; i < static_cast<int>(x.size()) - 1; ++i) {
          lipsh = (std::abs(f(x[i + 1]) - f(x[i]))) / (x[i + 1] - x[i]);
          if (lipsh > lipshM) {
            lipshM = lipsh;
            lipshm = lipshM + lipshM;
          }
        }
      }
      MPI_Send(&lipshm, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
  }
}

bool StronginParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

}  // namespace burykin_m_strongin
