#pragma once

#include <algorithm>
#include <vector>

namespace core_qwen3_forced_aligner {

// Exact port of Qwen3ForceAlignProcessor.fix_timestamp() at upstream
// QwenLM/Qwen3-ASR commit 7c6daf77a2421100f5fb066495372c00129d39ff.
inline void fix_timestamps(std::vector<int>& values) {
    const int n = (int)values.size();
    if (n == 0)
        return;

    std::vector<int> dp(n, 1);
    std::vector<int> parent(n, -1);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (values[j] <= values[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    const int max_len = *std::max_element(dp.begin(), dp.end());
    int idx = (int)(std::find(dp.begin(), dp.end(), max_len) - dp.begin());
    std::vector<bool> normal(n, false);
    while (idx != -1) {
        normal[idx] = true;
        idx = parent[idx];
    }

    int i = 0;
    while (i < n) {
        if (normal[i]) {
            i++;
            continue;
        }
        int j = i;
        while (j < n && !normal[j])
            j++;
        const int count = j - i;
        const bool have_left = i > 0;
        const bool have_right = j < n;
        const int left = have_left ? values[i - 1] : 0;
        const int right = have_right ? values[j] : 0;

        if (count <= 2) {
            for (int k = i; k < j; k++) {
                if (!have_left)
                    values[k] = right;
                else if (!have_right)
                    values[k] = left;
                else
                    values[k] = (k - (i - 1) <= j - k) ? left : right;
            }
        } else if (have_left && have_right) {
            const float step = (float)(right - left) / (float)(count + 1);
            for (int k = i; k < j; k++)
                values[k] = (int)(left + step * (float)(k - i + 1));
        } else {
            const int fill = have_left ? left : right;
            for (int k = i; k < j; k++)
                values[k] = fill;
        }
        i = j;
    }
}

} // namespace core_qwen3_forced_aligner
