float compute_reward(const std::vector<float>& state) {
    float reward = 0.0;

    float x = state[0];
    float theta = state[2];

    // Check if the pole has fallen
    if (std::abs(theta) > 0.5) {
        reward = -1.0;
    }
    else {
        // Reward for staying upright
        reward = 0.1;

        // Additional reward for staying near the center
        float center_reward = std::exp(-(std::pow(x, 2) / 10.0));
        reward += center_reward;
    }

    return reward;
}
