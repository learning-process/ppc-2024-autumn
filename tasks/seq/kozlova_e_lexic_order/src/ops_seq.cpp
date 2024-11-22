
#include "seq/kozlova_e_lexic_order/include/ops_seq.hpp"

#include <algorithm>

bool kozlova_e_lexic_order::StringComparator::pre_processing() {
  internal_order_test();

  auto* s1 = reinterpret_cast<char*>(taskData->inputs[0]);
  auto* s2 = reinterpret_cast<char*>(taskData->inputs[1]);

  str1 = std::string(s1);
  str2 = std::string(s2);
  return true;
}

bool kozlova_e_lexic_order::StringComparator::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2;
}

std::vector<int> kozlova_e_lexic_order::StringComparator::LexicographicallyOrdered() {
  int flag1 = 1;
  int flag2 = 1;
  std::vector<int> localres;
  std::string lowerStr1 = str1;
  std::transform(lowerStr1.begin(), lowerStr1.end(), lowerStr1.begin(), ::tolower);

  std::string lowerStr2 = str2;
  std::transform(lowerStr2.begin(), lowerStr2.end(), lowerStr2.begin(), ::tolower);
  for (size_t i = 0; i < lowerStr1.size() - 1; ++i) {
    if (lowerStr1[i] > lowerStr1[i + 1]) {
      flag1 = 0;
      break;
    }
  }

  for (size_t i = 0; i < lowerStr2.size() - 1; ++i) {
    if (lowerStr2[i] > lowerStr2[i + 1]) {
      flag2 = 0;
      break;
    }
  }
  localres.emplace_back(flag1);
  localres.emplace_back(flag2);
  return localres;
}

bool kozlova_e_lexic_order::StringComparator::run() {
  internal_order_test();
  res = LexicographicallyOrdered();
  return true;
}

bool kozlova_e_lexic_order::StringComparator::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = static_cast<int>(res[i]);
  }
  return true;
}
