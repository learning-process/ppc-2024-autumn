#include <vector>
#include <limits>

namespace nasedkin_e_matrix_column_max_value_seq {

std::vector<double> findMaxInColumns(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {};

    size_t cols = matrix[0].size();
    std::vector<double> maxValues(cols, std::numeric_limits<double>::lowest());

    for (const auto& row : matrix) {
        for (size_t j = 0; j < cols; ++j) {
            maxValues[j] = std::max(maxValues[j], row[j]);
        }
    }

    return maxValues;
}

}  // namespace nasedkin_e_matrix_column_max_value_seq
