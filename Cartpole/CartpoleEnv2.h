#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <vector>
using namespace torch;
using namespace std;
#define M_PI 3.14159265358979323846

class CartPoleEnv {
public:
    CartPoleEnv(float pole_length=0.5, float max_force=10.0);

    std::vector<float> reset();
    std::tuple<std::vector<float>, float, int> step(float action);
    std::vector<float> get_state();
    std::vector<float> get_observation_space();
    int get_action_space();

private:
    float m_pole_length;
    float m_max_force;
    float m_gravity;
    float m_masscart;
    float m_masspole;
    float m_total_mass;
    float m_length;
    float m_polemass_length;
    float m_force_mag;
    float m_tau;
    float m_theta_threshold_radians;
    float m_x_threshold;

    float m_state[4];
    int m_done;
    float m_reward;

    void _update_state(float force);
};

CartPoleEnv::CartPoleEnv(float pole_length, float max_force) :
    m_pole_length(pole_length),
    m_max_force(max_force),
    m_gravity(9.8),
    m_masscart(1.0),
    m_masspole(0.1),
    m_total_mass(m_masspole + m_masscart),
    m_length(2.0*m_pole_length),
    m_polemass_length(m_masspole * m_length),
    m_force_mag(9.0),
    m_tau(0.02),
    m_theta_threshold_radians(12 * 2 * M_PI / 360),
    m_x_threshold(2.4)
{
    reset();
}

std::vector<float> CartPoleEnv::reset() {
    // Reset the state of the environment
    float x = 0.0;
    float x_dot = 0.0;
    float theta = 0.0;
    float theta_dot = 0.0;
    m_state[0] = x;
    m_state[1] = x_dot;
    m_state[2] = theta;
    m_state[3] = theta_dot;

    m_done = false;
    m_reward = 0.0;

    return {x, x_dot, theta, theta_dot};
}

std::tuple<std::vector<float>, float, int> CartPoleEnv::step(float action) {
    // Apply the given action and update the state of the environment

    float force_directed = (action == 0) ? -m_force_mag : m_force_mag;

    float force = std::min(std::max(force_directed, -m_max_force), m_max_force);
    _update_state(force);

    // Compute the reward
    if (std::abs(m_state[0]) > m_x_threshold || std::abs(m_state[2]) > m_theta_threshold_radians) {
        m_done = 1;
        m_reward = -1.0;
        if(m_state[0]>m_x_threshold){
            cout << "Cartposition to far!"<< endl;
        }
        else{
            cout << "Pole fell!"<< endl;
        }
    } else {
        
        m_reward = 1.0;
    }

    return {{m_state[0], m_state[1], m_state[2], m_state[3]}, m_reward, m_done};
}

std::vector<float> CartPoleEnv::get_state() {
    // Return the current state of the environment
    return {m_state[0], m_state[1], m_state[2], m_state[3]};
}

std::vector<float> CartPoleEnv::get_observation_space() {
    //std::vector<float>
    // Return the observation space of the environment
    return {-m_x_threshold, m_x_threshold, -m_theta_threshold_radians, m_theta_threshold_radians};;
}
    int CartPoleEnv::get_action_space() {
    // Return the action space of the environment
    return 2; // two possible actions: move left or right
}

//void CartPoleEnv::update_state(float action) {
//    // Convert the action from a tensor to an integer
//    //int64_t int_action = action.item<int64_t>();
//
//    // Apply the action to the environment
//    CartPoleEnv::step(action);
//
//    // Update the state variables
//    m_x = m_env.x();
//    m_x_dot = m_env.x_dot();
//    m_theta = m_env.theta();
//    m_theta_dot = m_env.theta_dot();
//
//    // Check if the episode is done
//    m_done = m_env.done();
//}
void CartPoleEnv::_update_state(float force) {
    // Compute the derivatives of the state variables
    float cos_theta = std::cos(m_state[2]);
    float sin_theta = std::sin(m_state[2]);
    float temp = (force + m_polemass_length * m_state[3] * m_state[3] * sin_theta) / m_total_mass;
    float theta_acc = (m_gravity * sin_theta - cos_theta * temp) / (m_length * (4.0/3.0 - m_masspole * cos_theta * cos_theta / m_total_mass));
    float x_acc = temp - m_polemass_length * theta_acc * cos_theta / m_total_mass;
    float x_dot_dot = x_acc;
    float x_dot = m_state[1] + m_tau * x_dot_dot;
    float x = m_state[0] + m_tau * x_dot;
    float theta_dot_dot = theta_acc;
    float theta_dot = m_state[3] + m_tau * theta_dot_dot;
    float theta = m_state[2] + m_tau * theta_dot;

    // Update the state variables
    m_state[0] = x;
    m_state[1] = x_dot;
    m_state[2] = theta;
    m_state[3] = theta_dot;
}

