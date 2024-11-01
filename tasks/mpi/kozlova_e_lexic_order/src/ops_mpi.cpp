// Copyright 2023 Nesterov Alexander
#include "mpi/kozlova_e_lexic_order/include/ops_mpi.hpp"

#include <random>
#include <string>
#include <vector>

std::vector<int> LexicographicallyOrdered(const std::string& str1, const std::string& str2) {
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

bool kozlova_e_lexic_order_mpi::StringComparatorSeq::pre_processing() {
  internal_order_test();

  auto* s1 = reinterpret_cast<char*>(taskData->inputs[0]);
  auto* s2 = reinterpret_cast<char*>(taskData->inputs[1]);

  str1 = std::string(s1);
  str2 = std::string(s2);
  return true;
}

bool kozlova_e_lexic_order_mpi::StringComparatorSeq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 2;
}

bool kozlova_e_lexic_order_mpi::StringComparatorSeq::run() {
  internal_order_test();
  res.resize(2);
  if (str1.empty()) res[0] = 1;
  if (str2.empty())
      res[1] = 1;
  else
      res = LexicographicallyOrdered(str1, str2);
  return true;
}

bool kozlova_e_lexic_order_mpi::StringComparatorSeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = static_cast<int>(res[i]);
  }
  return true;
}

bool kozlova_e_lexic_order_mpi::StringComparatorMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* s1 = reinterpret_cast<char*>(taskData->inputs[0]);
    auto* s2 = reinterpret_cast<char*>(taskData->inputs[1]);
    input_strings.resize(2);
    input_strings[0] = std::string(s1);
    input_strings[1] = std::string(s2);
  } else {
    input_strings.resize(2);
  }

  for (int i = 0; i < 2; i++) boost::mpi::broadcast(world, input_strings[i], 0);
  res.resize(2, 0);
  return true;
}

bool kozlova_e_lexic_order_mpi::StringComparatorMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] == 2;
  }
  return true;
}

bool kozlova_e_lexic_order_mpi::StringComparatorMPI::run() {
  internal_order_test();

  std::vector<int> local_res(2, 1);
  int len1 = input_strings[0].size();
  int len2 = input_strings[1].size();
  int delta1 = (len1 + world.size() - 1) / world.size();
  int delta2 = (len2 + world.size() - 1) / world.size();
  int start1 = world.rank() * delta1;
  int end1 = std::min(start1 + delta1, len1);
  int start2 = world.rank() * delta2;
  int end2 = std::min(start2 + delta2, len2);

  std::string local_string1 = (start1 < len1) ? input_strings[0].substr(start1, end1 - start1) : "";
  std::string local_string2 = (start2 < len2) ? input_strings[1].substr(start2, end2 - start2) : "";

  if (!local_string1.empty()) {
    local_res[0] = LexicographicallyOrdered(local_string1, local_string2)[0];
  }
  if (!local_string2.empty()) {
    local_res[1] = LexicographicallyOrdered(local_string1, local_string2)[1];
  }

  if (world.rank() < world.size() - 1) {
    if (end1 > 0 && end1 < len1) {
      char last_char1 = std::tolower(input_strings[0][end1 - 1]);
      char first_char1_next = std::tolower(input_strings[0][end1]);
      if (last_char1 > first_char1_next) {
        local_res[0] = 0;
      }
    }
    if (end2 > 0 && end2 < len2) {
      char last_char2 = std::tolower(input_strings[1][end2 - 1]);
      char first_char2_next = std::tolower(input_strings[1][end2]);
      if (last_char2 > first_char2_next) {
        local_res[1] = 0;
      }
    }
  }

  std::vector<int> global_results(2 * world.size());
  boost::mpi::gather(world, local_res.data(), local_res.size(), global_results.data(), 0);

  if (world.rank() == 0) {
    int is_ordered1 = 1;
    int is_ordered2 = 1;
    for (int i = 0; i < world.size(); ++i) {
      if (global_results[i * 2] == 0) is_ordered1 = 0;
      if (global_results[i * 2 + 1] == 0) is_ordered2 = 0;
    }
    res[0] = is_ordered1;
    res[1] = is_ordered2;
  }
  return true;
}

bool kozlova_e_lexic_order_mpi::StringComparatorMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < res.size(); i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
