#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <cmath>
#define M_PI 3.14159265358979323846
class CartPoleEnv {
public:
    CartPoleEnv(float gravity = 9.8f, float masscart = 1.0f, float masspole = 0.1f, float length = 0.5f, float force_mag = 10.0f,
                float tau = 0.02f, float theta_threshold_radians = 12.0f * 2.0f * M_PI / 360.0f, float x_threshold = 2.4f)
        : gravity(gravity), masscart(masscart), masspole(masspole), total_mass(masscart + masspole),
          length(length), polemass_length(masspole * length), force_mag(force_mag), tau(tau),
          theta_threshold_radians(theta_threshold_radians), x_threshold(x_threshold), generator(std::chrono::system_clock::now().time_since_epoch().count()) {
        reset();
    }

    std::vector<float> reset() {
        x = 0.0f;
        x_dot = 0.0f;
        theta = 0.0f;
        theta_dot = 0.0f;
        std::uniform_real_distribution<float> uniform(-0.05f, 0.05f);
        float init_state[4] = {uniform(generator), 0.0f, uniform(generator), 0.0f};
        return std::vector<float>(init_state, init_state + 4);
    }

    std::tuple<std::vector<float>, float, bool> step(int action) {
        float force = action == 1 ? force_mag : -force_mag;
        float costheta = cos(theta);
        float sintheta = sin(theta);
        float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        float thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
        float xacc = temp - polemass_length * thetaacc * costheta / total_mass;
        x += tau * x_dot;
        x_dot += tau * xacc;
        theta += tau * theta_dot;
        theta_dot += tau * thetaacc;
        bool done = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;
        float reward = done ? 0.0f : 1.0f;
        return std::make_tuple(std::vector<float>{x, x_dot, theta, theta_dot}, reward, done);
    }

private:
    float gravity;
    float masscart;
    float masspole;
    float total_mass;
    float length;
    float polemass_length;
    float force_mag;
    float tau;
    float theta_threshold_radians;
    float x_threshold;
    float x;
    float x_dot;
    float theta;
    float theta_dot;
    std::mt19937 generator;
};
