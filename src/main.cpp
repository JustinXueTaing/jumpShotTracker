#include <Arduino.h>
#include <Wire.h>
#include "bmi270_init.h"
#include "bmi270_hal.h"

extern "C" {
  #include "bmi270.h"
  #include "bmi2_defs.h"
}

uint8_t i2c_addr = 0x68;
struct bmi2_dev dev = {
  .intf_ptr = &i2c_addr,
  .intf = BMI2_I2C_INTF,
  .read = bmi270_i2c_read,
  .write = bmi270_i2c_write,
  .delay_us = bmi270_delay_us
};

void setup() {
  Serial.begin(115200);
  delay(100);
  Wire.begin();
  InitResult result = initBMI270(dev);
    if (result != InitResult::SUCCESS) {
      Serial.println("Initialization failed");  
        while(true) {}
    }
    
}

void loop() {
  // put your main code here, to run repeatedly:
}
