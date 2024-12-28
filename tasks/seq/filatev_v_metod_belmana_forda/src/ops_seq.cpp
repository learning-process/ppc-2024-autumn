// Filatev Vladislav Metod Belmana Forda
#include "seq/filatev_v_metod_belmana_forda/include/ops_seq.hpp"

bool filatev_v_metod_belmana_forda_seq::MetodBelmanaForda::validation() {
  internal_order_test();
  int n_ = taskData->inputs_count[0];
  int m_ = taskData->inputs_count[1];
  int start_ = taskData->inputs_count[2];
  int n_o = taskData->outputs_count[0];
  return n_ > 0 && m_ > 0 && m_ <= (n_ - 1) * n_ && start_ >= 0 && start_ < n_ && n_o == n_;
}

bool filatev_v_metod_belmana_forda_seq::MetodBelmanaForda::pre_processing() {
  internal_order_test();

  this->n = taskData->inputs_count[0];
  this->m = taskData->inputs_count[1];
  this->start = taskData->inputs_count[2];

  return true;
}

bool filatev_v_metod_belmana_forda_seq::MetodBelmanaForda::run() {
  internal_order_test();

  int inf = std::numeric_limits<int>::max();
  d.assign(n, inf);
  d[start] = 0;

  auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
  this->Adjncy.assign(temp, temp + m);
  temp = reinterpret_cast<int*>(taskData->inputs[1]);
  this->Xadj.assign(temp, temp + n + 1);
  temp = reinterpret_cast<int*>(taskData->inputs[2]);
  this->Eweights.assign(temp, temp + m);

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

bool filatev_v_metod_belmana_forda_seq::MetodBelmanaForda::post_processing() {
  internal_order_test();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(d.begin(), d.end(), output_data);
  return true;
}
