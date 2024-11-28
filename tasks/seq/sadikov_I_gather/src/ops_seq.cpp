#include <iostream>
#include <random>

#include "seq/sadikov_I_gather/include/sq_task.h"

sadikov_I_gather_seq::SumByRowsTask::SumByRowsTask(std::shared_ptr<ppc::core::TaskData> taskData)
    : Task(std::move(taskData)) {}

bool sadikov_I_gather_seq::SumByRowsTask::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool sadikov_I_gather_seq::SumByRowsTask::pre_processing() {
  internal_order_test();
  m_rowsCount = static_cast<size_t>(taskData->inputs_count[0]);
  m_columnsCount = static_cast<size_t>(taskData->inputs_count[1]);
  m_matrix.reserve(m_rowsCount * m_columnsCount);
  auto *tmpPtr = reinterpret_cast<int *>(taskData->inputs[0]);
  for (size_t i = 0; i < m_columnsCount * m_rowsCount; ++i) {
    m_matrix.emplace_back(tmpPtr[i]);
  }
  m_sum = std::vector<int>(m_rowsCount, 0);
  return true;
}

bool sadikov_I_gather_seq::SumByRowsTask::run() {
  internal_order_test();
  Calculate(m_rowsCount);
  return true;
}

bool sadikov_I_gather_seq::SumByRowsTask::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < m_rowsCount; ++i) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = m_sum[i];
  }
  return true;
}

void sadikov_I_gather_seq::SumByRowsTask::Calculate(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    m_sum[i] = std::accumulate(m_matrix.begin() + i * m_columnsCount, m_matrix.begin() + (i + 1) * m_columnsCount, 0);
  }
}

std::shared_ptr<ppc::core::TaskData> sadikov_I_gather_seq::CreateTaskData(std::vector<int> &InV,
                                                                          const std::vector<int> &CeV,
                                                                          std::vector<int> &OtV) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(InV.data()));
  taskData->inputs_count.emplace_back(CeV[0]);
  taskData->inputs_count.emplace_back(CeV[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(OtV.data()));
  taskData->outputs_count.emplace_back(OtV.size());
  return taskData;
}
