#pragma once

#include <cmath>
#include <array>
#include <algorithm>

/**
 * @brief Оптимизированный класс Parametric Memory Gate (PMG) для одного канала.
 */
class ParametricMemoryGate {
private:
    float base = 4.0f;
    float shift = -1.0f;
    const float eps = 1e-7f;

public:
    ParametricMemoryGate() = default;

    /**
     * @brief Инициализация параметров геометрии гейта (задаются после обучения в PyTorch)
     */
    void init(float learned_base, float learned_shift) {
        if (learned_base > 1.0f) {
            base = learned_base;
        }
        shift = learned_shift;
    }

    /**
     * @brief Высокоскоростной пошаговый инференс одного значения
     */
    float filter_step(float x_t) {
        // Защита от "битых" данных с датчиков (Hardware Fault Protection)
        if (std::isnan(x_t) || std::isinf(x_t)) {
            return 0.0f; 
        }

        // Ограничение экспоненты в соответствии с оригинальной архитектурой (-20.0, 20.0)
        float power = x_t + shift;
        power = std::max(-20.0f, std::min(20.0f, power));

        // Основная математическая формула PMG
        float base_pow = std::powf(base, power);
        float gate = base_pow / (1.0f + base_pow);

        // Защита от аппаратного округления до абсолютного нуля или единицы
        return std::max(eps, std::min(1.0f - eps, gate));
    }
};

/**
 * @brief Многоканальный параллельный PMG-фильтр прямого действия.
 * @tparam CHANNELS Количество независимых осей/каналов телеметрии (например, 3 для X,Y,Z или 1 для RSSI)
 */
template <size_t CHANNELS>
class PureParallelPMG {
private:
    std::array<ParametricMemoryGate, CHANNELS> filters;

public:
    PureParallelPMG() = default;

    /**
     * @brief Пакетная инициализация коэффициентов осей
     */
    void setup_channels(const std::array<float, CHANNELS>& bases, const std::array<float, CHANNELS>& shifts) {
        for (size_t i = 0; i < CHANNELS; ++i) {
            filters[i].init(bases[i], shifts[i]);
        }
    }

    /**
     * @brief Пошаговый расчет всего вектора данных телеметрии (вызывается в цикле 100Гц - 8кГц)
     * @param input_vector Массив сырых зашумленных данных с датчиков
     * @return std::array<float, CHANNELS> Идеально очищенный вектор данных
     */
    std::array<float, CHANNELS> update(const std::array<float, CHANNELS>& input_vector) {
        std::array<float, CHANNELS> output_vector;
        
        // Независимая параллельная обработка каналов в векторном стиле
        for (size_t i = 0; i < CHANNELS; ++i) {
            output_vector[i] = filters[i].filter_step(input_vector[i]);
        }
        
        return output_vector;
    }
};
