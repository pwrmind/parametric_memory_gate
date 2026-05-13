#pragma once
#include <cmath>
#include <algorithm>

class InertialParametricGate {
private:
    float base = 4.0f;
    float shift = -1.0f;
    float x_prev = 0.0f;       // Встроенная память предыдущего состояния
    bool is_initialized = false; // Флаг для корректного захвата первой точки
    const float eps = 1e-7f;

public:
    InertialParametricGate() = default;

    void init(float learned_base, float learned_shift) {
        if (learned_base > 1.0f) {
            base = learned_base;
        }
        shift = learned_shift;
        reset();
    }

    void reset() {
        x_prev = 0.0f;
        is_initialized = false;
    }

    /**
     * @brief Автономный шаг фильтрации. Физика дельты скрыта внутри функции.
     * @param x_t Текущее сырое значение телеметрии (например, координата GPS)
     * @return Отфильтрованный импульс динамического отклика
     */
    float filter_step(float x_t) {
        if (std::isnan(x_t) || std::isinf(x_t)) {
            return 0.0f;
        }

        // При первом запуске инициализируем память текущей точкой,
        // чтобы избежать ложного скачка дельты из абсолютного нуля
        if (!is_initialized) {
            x_prev = x_t;
            is_initialized = true;
            return 0.0f; 
        }

        // Рассчитываем внутреннюю дельту скорости изменения
        float delta_x = x_t - x_prev;
        
        // Перезаписываем память для следующего такта
        x_prev = x_t;

        // Расчет оригинальной экспоненциальной геометрии PMG
        float power = delta_x + shift;
        power = std::max(-20.0f, std::min(20.0f, power));

        float base_pow = std::powf(base, power);
        float gate = base_pow / (1.0f + base_pow);

        return std::max(eps, std::min(1.0f - eps, gate));
    }
};
