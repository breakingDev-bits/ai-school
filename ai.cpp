// Збірка:
//   Linux:   g++ -O2 -shared -fPIC nn.cpp -o libnn.so
//   macOS:   g++ -O2 -shared -fPIC nn.cpp -o libnn.dylib
//   Windows: g++ -O2 -shared nn.cpp -o nn.dll

/*
Для Windows використовуйте MinGW, а не MSVC. Я не враховував поведінки MSVC.
Під Linux використовуйте Clang++/G++. Під WSL найкраще G++.
Під macOS не пробував. Якщо ви на ній, найімовірніше, найкраще Clang++.

Важливо! Без -O3, я не гарантирую правильное поведения для CTypes и чего другого. Только -O2/-O1
*/

#include <cmath>
#include <random>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

struct StepData {
    double prediction;
    double error;
    double w0;
    double w1;
    double bias;
};

std::vector<double> random_weight(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.1, 1.0);

    std::vector<double> weights(size);
    for (size_t i = 0; i < size; ++i) {
        weights[i] = dis(gen);
    }

    return weights;
}

std::vector<StepData> study_process(double x, double y, double target_result,
                                    double learning_rate, double eps, int max_steps) {
    std::vector<double> weights = random_weight(2);
    double bias = 0.5;

    std::vector<StepData> history;
    history.reserve(max_steps);

    for (int i = 0; i < max_steps; i++) {
        double prediction = (x * weights[0]) + (y * weights[1]) + bias;
        double error = target_result - prediction;

        history.push_back({prediction, error, weights[0], weights[1], bias});

        if (std::abs(error) < eps) {
            break;
        }

        weights[0] += learning_rate * error * x;
        weights[1] += learning_rate * error * y;
        bias += learning_rate * error;
    }

    return history;
}

EXPORT int nn_train(double x, double y, double target, double lr, double eps,
                    int max_steps, StepData* out) {
    std::vector<StepData> history = study_process(x, y, target, lr, eps, max_steps);
    for (size_t i = 0; i < history.size(); ++i) {
        out[i] = history[i];
    }
    return static_cast<int>(history.size());
}

EXPORT double nn_predict(double x, double y, double w0, double w1, double bias) {
    return x * w0 + y * w1 + bias;
}