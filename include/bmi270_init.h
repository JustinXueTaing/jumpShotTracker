#pragma once
#include <Arduino.h>

extern "C" {
    #include "bmi270.h"
    #include "bmi2_defs.h"
}

enum class InitResult {
    SUCCESS,
    SENSOR_NOT_DETECTED,
    REGISTER_VERIFY_FAIL,
    FIFO_FAIL,
};

InitResult initBMI270(bmi2_dev& sensor);