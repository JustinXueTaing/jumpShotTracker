#include <Arduino.h>
#include <Wire.h>
#include "bmi2_defs.h"

BMI2_INTF_RETURN_TYPE bmi270_i2c_read(uint8_t reg_addr, 
                                      uint8_t *reg_data, 
                                      uint32_t len, 
                                      void *intf_ptr)
{ 
  uint8_t addr = *((uint8_t *) intf_ptr);
  Wire.beginTransmission(addr);
  Wire.write(reg_addr);
  if (Wire.endTransmission(false) != 0) {
    return BMI2_E_COM_FAIL;
  }
  Wire.requestFrom(addr, len);
  for (int i = 0; i < len; i++) {
    reg_data[i] = Wire.read();
  }
  return BMI2_OK;
}

BMI2_INTF_RETURN_TYPE bmi270_i2c_write(uint8_t reg_addr,
                                       const uint8_t *reg_data,
                                       uint32_t len,
                                       void *intf_ptr)
{
  uint8_t addr = *((uint8_t *) intf_ptr);
  Wire.beginTransmission(addr);
  Wire.write(reg_addr);
  for (int i = 0; i < len; i++)
  {
    Wire.write(reg_data[i]);
  }
  if (Wire.endTransmission(true) != 0) {
    return BMI2_E_COM_FAIL;
  }

  return BMI2_OK;
}

void bmi270_delay_us(uint32_t period, void *intf_ptr)
{
  delayMicroseconds(period);
}