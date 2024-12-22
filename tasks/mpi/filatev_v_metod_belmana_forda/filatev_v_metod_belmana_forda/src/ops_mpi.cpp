// Filatev Vladislav Metod Belmana Forda
#include "mpi/filatev_v_metod_belmana_forda/include/ops_mpi.hpp"

#include <boost/serialization/vector.hpp>
#include <vector>

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    int n_ = taskData->inputs_count[0];
    int m_ = taskData->inputs_count[1];
    int start_ = taskData->inputs_count[2];
    int n_o = taskData->outputs_count[0];
    return n_ > 0 && m_ > 0 && m_ <= (n_ - 1) * n_ && start_ >= 0 && start_ < n_ && n_o == n_;
  }
  return true;
}

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    this->n = taskData->inputs_count[0];
    this->m = taskData->inputs_count[1];
    this->start = taskData->inputs_count[2];

    auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
    this->Adjncy.assign(temp, temp + m);
    temp = reinterpret_cast<int*>(taskData->inputs[1]);
    this->Xadj.assign(temp, temp + n + 1);
    temp = reinterpret_cast<int*>(taskData->inputs[2]);
    this->Eweights.assign(temp, temp + m);
    d.resize(n);
  }
  return true;
}

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, m, 0);
  boost::mpi::broadcast(world, start, 0);

  int inf = std::numeric_limits<int>::max();
  d.assign(n, inf);
  d[start] = 0;

  if (world.size() == 1 || world.size() > n) {
    bool stop = true;
    for (int i = 0; i < n; i++) {
      stop = true;
      for (int v = 0; v < n; v++) {
        for (int t = Xadj[v]; t < Xadj[v + 1]; t++) {
          if (d[v] < inf && d[Adjncy[t]] > d[v] + Eweights[t]) {
            d[Adjncy[t]] = d[v] + Eweights[t];
            stop = false;
          }
        }
      }
      if (stop) {
        break;
      }
    }

    if (!stop) {
      d.assign(n, -inf);
    }

    return true;
  }

  int delta = n / world.size();
  int ost = n % world.size();

  int rank = world.rank();

  int start_v = (wrank < ost) ? (delta + 1) * rank : (delta + 1) * ost + (rank - ost) * delta;
  int stop_v = (rank < ost) ? (delta + 1) * (rank + 1) : (delta + 1) * ost + (rank - ost + 1) * delta;

  if (world.rank() != 0) {
    Xadj.resize(n + 1);
  }
  boost::mpi::broadcast(world, Xadj.data(), n + 1, 0);

  std::vector<int> distribution(world.size(), 0);
  std::vector<int> displacement(world.size(), 0);
  int prev = Xadj[(ost == 0) ? delta : delta + 1];
  distribution[0] = prev;
  for (int i = 1; i < world.size(); i++) {
    int teck = 0;
    if (i < ost) {
      teck = Xadj[(delta + 1) * (i + 1)];
    } else {
      teck = Xadj[(delta + 1) * ost + delta * (i + 1 - ost)];
    }
    distribution[i] = teck - prev;
    displacement[i] = displacement[i - 1] + distribution[i - 1];
    prev = teck;
  }

  int local_size = distribution[world.rank()];
  std::vector<int> local_Adjncy(local_size);
  std::vector<int> local_Eweights(local_size);
  boost::mpi::scatterv(world, Adjncy.data(), distribution, displacement, local_Adjncy.data(), local_size, 0);
  boost::mpi::scatterv(world, Eweights.data(), distribution, displacement, local_Eweights.data(), local_size, 0);

  std::vector<int> local_d(n);

  bool stop = true;
  for (int i = 0; i < n; i++) {
    boost::mpi::broadcast(world, d, 0);
    bool local_stop = true;
    for (int v = start_v; v < stop_v; v++) {
      if (v > (int)Xadj.size() - 2) continue;
      for (int t = Xadj[v]; t < Xadj[v + 1]; t++) {
        int l_posit = t - Xadj[start_v];
        if (d[v] < inf && d[local_Adjncy[l_posit]] > d[v] + local_Eweights[l_posit]) {
          d[local_Adjncy[l_posit]] = d[v] + local_Eweights[l_posit];
          local_stop = false;
        }
      }
    }
    std::copy(d.begin(), d.end(), local_d.begin());
    reduce(world, local_d, d, boost::mpi::minimum<int>(), 0);
    reduce(world, local_stop, stop, boost::mpi::minimum<int>(), 0);
    boost::mpi::broadcast(world, stop, 0);
    if (stop) {
      break;
    }
  }

  if (world.rank() == 0 && !stop) {
    d.assign(n, -inf);
  }

  return true;
}

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(d.begin(), d.end(), output_data);
  }
  return true;
}

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaSeq::validation() {
  internal_order_test();
  int n_ = taskData->inputs_count[0];
  int m_ = taskData->inputs_count[1];
  int start_ = taskData->inputs_count[2];
  int n_o = taskData->outputs_count[0];
  return n_ > 0 && m_ > 0 && m_ <= (n_ - 1) * n_ && start_ >= 0 && start_ < n_ && n_o == n_;
}

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaSeq::pre_processing() {
  internal_order_test();

  this->n = taskData->inputs_count[0];
  this->m = taskData->inputs_count[1];
  this->start = taskData->inputs_count[2];

  auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
  this->Adjncy.assign(temp, temp + m);
  temp = reinterpret_cast<int*>(taskData->inputs[1]);
  this->Xadj.assign(temp, temp + n + 1);
  temp = reinterpret_cast<int*>(taskData->inputs[2]);
  this->Eweights.assign(temp, temp + m);

  this->d.resize(n);

  return true;
}

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaSeq::run() {
  internal_order_test();

  int inf = std::numeric_limits<int>::max();
  d.assign(n, inf);
  d[start] = 0;

  bool stop = true;
  for (int i = 0; i < n; i++) {
    stop = true;
    for (int v = 0; v < n; v++) {
      for (int t = Xadj[v]; t < Xadj[v + 1]; t++) {
        if (d[v] < inf && d[Adjncy[t]] > d[v] + Eweights[t]) {
          d[Adjncy[t]] = d[v] + Eweights[t];
          stop = false;
        }
      }
    }
    if (stop) {
      break;
    }
  }

  if (!stop) {
    d.assign(n, -inf);
  }

  return true;
}

bool filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaSeq::post_processing() {
  internal_order_test();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(d.begin(), d.end(), output_data);
  return true;
}
