#include "ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace nasedkin_e_matrix_column_max_value_seq {

    std::vector<int> FindColumnMaxSequential(const std::vector<std::vector<int>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            return {};
        }
        std::vector<int> max_values(matrix[0].size(), std::numeric_limits<int>::min());
        for (const auto& row : matrix) {
            for (size_t col = 0; col < row.size(); ++col) {
                max_values[col] = std::max(max_values[col], row[col]);
            }
        }
        return max_values;
    }

}
