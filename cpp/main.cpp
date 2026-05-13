#include <iostream>
#include "PureParallelPMG.hpp"

int main() {
    // 1. Создаем фильтр для 3 осей (X, Y, Z)
    PureParallelPMG<3> telemetry_filter;

    // 2. Загружаем оптимальные параметры, полученные в ходе нашего Python-теста
    std::array<float, 3> learned_bases  = {5.3881f, 2.7058f, 1.6814f};
    std::array<float, 3> learned_shifts = {-0.1106f, -1.0408f, 2.0181f};
    
    telemetry_filter.setup_channels(learned_bases, learned_shifts);

    // 3. Эмуляция полетного цикла (Поступление зашумленного пакета телеметрии)
    // Допустим, ось Z плавно растет, а по X и Y прилетел шум
    std::array<float, 3> raw_telemetry = {0.54f, 0.41f, 0.12f}; 

    std::array<float, 3> clean_telemetry = telemetry_filter.update(raw_telemetry);

    // Вывод в консоль
    std::cout << "--- Тест работы PMG на C++ ---" << std::endl;
    std::cout << "Сырые данные:  X: " << raw_telemetry[0] << " | Y: " << raw_telemetry[1] << " | Z: " << raw_telemetry[2] << std::endl;
    std::cout << "PMG Фильтр:    X: " << clean_telemetry[0] << " | Y: " << clean_telemetry[1] << " | Z: " << clean_telemetry[2] << std::endl;

    return 0;
}
