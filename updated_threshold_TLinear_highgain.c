#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"

#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "lwip/inet.h"
#include "lwip/err.h"

static const char *TAG = "WIFI_THERMAL_AP";

/* ---------------- Wi-Fi SoftAP ---------------- */
#define AP_SSID                 "ESP32_THERMAL"
#define AP_PASS                 "ThermalTest2026"
#define AP_CHANNEL              6
#define AP_MAX_CONN             4

/*
 * We are using a Macbook.
 * Keep this as the Mac IP on the ESP32 Wi-Fi.
 * If the Mac later gets a different IP, update it.
 */
#define SERVER_IP               "192.168.4.2"
#define SERVER_PORT             5001

/* ---------------- Pin Mapping ----------------
   ESP32                FLIR Lepton Breakout Board V2
   GPIO21  -----------> SDA
   GPIO22  -----------> SCL
   GPIO18  -----------> SPI_CLK
   GPIO19  <----------- SPI_MISO
   GPIO5   -----------> SPI_CS
   GPIO27  <----------> VSYNC (optional, unused here)
   3V3     -----------> VIN
   GND     -----------> GND
------------------------------------------------ */

/* ---------- I2C ---------- */
#define I2C_PORT_NUM            0
#define I2C_SDA_GPIO            21
#define I2C_SCL_GPIO            22
#define I2C_SCL_SPEED_HZ        400000
#define LEPTON_I2C_ADDR         0x2A

/* ---------- SPI / VoSPI ---------- */
#define LEPTON_SPI_HOST         SPI2_HOST
#define LEPTON_SPI_SCK_GPIO     18
#define LEPTON_SPI_MISO_GPIO    19
#define LEPTON_SPI_MOSI_GPIO    -1
#define LEPTON_SPI_CS_GPIO      5
#define LEPTON_VSYNC_GPIO       27

#define LEPTON_SPI_CLOCK_HZ     16000000
#define LEPTON_SPI_MODE         3

/* ---------- Frame format ---------- */
#define LEPTON_PACKET_SIZE          164
#define LEPTON_PACKET_HEADER        4
#define LEPTON_LINE_PIXELS          80
#define LEPTON_FRAME_LINES          60
#define LEPTON_FRAME_PIXELS         (LEPTON_LINE_PIXELS * LEPTON_FRAME_LINES)
#define LEPTON_FRAME_BYTES          (LEPTON_FRAME_PIXELS * sizeof(uint16_t))
#define LEPTON_MAX_RESYNC_PACKETS   5000

/* ---------- Threshold ---------- */
/*
 * Once TLinear is active, this is in actual Celsius and converted to counts.
 * This is only for the "super hot" flag, not for body-temperature display.
 */
#define HOT_THRESHOLD_C             60.0f
#define HOT_THRESHOLD_RAW_FALLBACK  14000u

/* ---------- Header flags ---------- */
#define MODE_FLAG_TEST_PATTERN      0x01
#define MODE_FLAG_HIGH_GAIN         0x02
#define MODE_FLAG_TLINEAR           0x04
#define MODE_FLAG_TLINEAR_0_01K     0x08

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t width;
    uint16_t height;
    uint32_t frame_id;
    uint32_t payload_bytes;
    uint16_t max_value;
    uint16_t hot_threshold;
    uint8_t  hot_flag;
    uint8_t  mode_flags;
    uint16_t reserved;
} thermal_header_t;

typedef struct {
    uint64_t timestamp_ms;
    uint32_t frame_id;
    uint16_t max_raw;
    uint16_t min_raw;
    int max_x;
    int max_y;
    int min_x;
    int min_y;
    float avg_raw;
    float left_avg;
    float center_avg;
    float right_avg;
} thermal_summary_t;

/* ---------- Lepton CCI/TWI register map ---------- */
#define LEP_REG_POWER                0x0000
#define LEP_REG_STATUS               0x0002
#define LEP_REG_COMMAND              0x0004
#define LEP_REG_DATA_LENGTH          0x0006
#define LEP_REG_DATA0                0x0008

#define LEP_CMD_PROTECTION_BIT       0x4000

#define LEP_CMD_TYPE_GET             0x0000
#define LEP_CMD_TYPE_SET             0x0001
#define LEP_CMD_TYPE_RUN             0x0002

#define LEP_MOD_AGC                  0x0100
#define LEP_MOD_SYS                  0x0200
#define LEP_MOD_RAD                  0x0E00

#define LEP_AGC_ENABLE_BASE          0x0000
#define LEP_SYS_RUN_FFC_BASE         0x0040
#define LEP_SYS_FFC_STATUS_BASE      0x0044
#define LEP_SYS_GAIN_MODE_BASE       0x0048

#define LEP_RAD_ENABLE_BASE          0x0010
#define LEP_RAD_FLUX_LINEAR_BASE     0x00BC
#define LEP_RAD_TLINEAR_ENABLE_BASE  0x00C0
#define LEP_RAD_TLINEAR_RES_BASE     0x00C4
#define LEP_RAD_TLINEAR_AUTO_BASE    0x00C8

#define LEP_AGC_DISABLE              0
#define LEP_AGC_ENABLE               1

#define LEP_SYS_GAIN_MODE_HIGH       0
#define LEP_SYS_GAIN_MODE_LOW        1
#define LEP_SYS_GAIN_MODE_AUTO       2

#define LEP_RAD_DISABLE              0
#define LEP_RAD_ENABLE               1

#define LEP_RAD_RESOLUTION_0_1       0
#define LEP_RAD_RESOLUTION_0_01      1

#define LEP_SYS_STATUS_READY         0
#define LEP_SYS_STATUS_BUSY          1

typedef struct {
    uint16_t sceneEmissivity;  // 3.13 fixed-point, 8192 = 100%
    uint16_t TBkgK;            // Kelvin x100
    uint16_t tauWindow;        // 3.13 fixed-point
    uint16_t TWindowK;         // Kelvin x100
    uint16_t tauAtm;           // 3.13 fixed-point
    uint16_t TAtmK;            // Kelvin x100
    uint16_t reflWindow;       // 3.13 fixed-point
    uint16_t TReflK;           // Kelvin x100
} lepton_flux_linear_params_t;

static i2c_master_bus_handle_t i2c_bus = NULL;
static i2c_master_dev_handle_t lepton_i2c_dev = NULL;
static spi_device_handle_t lepton_spi = NULL;

static uint16_t frame_buffer[LEPTON_FRAME_PIXELS];
static uint32_t g_frame_id = 0;

static bool g_high_gain_enabled = false;
static bool g_tlinear_enabled = false;
static bool g_tlinear_0_01k = false;

/* ---------------- Utility ---------------- */

static int send_all(int sock, const void *data, size_t len)
{
    const uint8_t *ptr = (const uint8_t *)data;
    size_t total_sent = 0;

    while (total_sent < len) {
        int sent = send(sock, ptr + total_sent, len - total_sent, 0);
        if (sent < 0) {
            ESP_LOGE(TAG, "send failed: errno=%d", errno);
            return -1;
        }
        total_sent += (size_t)sent;
    }

    return 0;
}

static uint16_t emissivity_to_q13(float e)
{
    if (e < 0.01f) e = 0.01f;
    if (e > 1.0f)  e = 1.0f;
    return (uint16_t)(e * 8192.0f + 0.5f);
}

static uint16_t celsius_to_kelvin_x100(float temp_c)
{
    float k = (temp_c + 273.15f) * 100.0f;
    if (k < 0.0f) k = 0.0f;
    if (k > 65535.0f) k = 65535.0f;
    return (uint16_t)(k + 0.5f);
}

static uint16_t celsius_to_tlinear_counts(float temp_c)
{
    if (!g_tlinear_enabled) {
        return HOT_THRESHOLD_RAW_FALLBACK;
    }

    if (g_tlinear_0_01k) {
        float v = (temp_c + 273.15f) * 100.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 65535.0f) v = 65535.0f;
        return (uint16_t)(v + 0.5f);
    } else {
        float v = (temp_c + 273.15f) * 10.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 65535.0f) v = 65535.0f;
        return (uint16_t)(v + 0.5f);
    }
}

static uint16_t get_hot_threshold_counts(void)
{
    if (g_tlinear_enabled) {
        return celsius_to_tlinear_counts(HOT_THRESHOLD_C);
    }
    return HOT_THRESHOLD_RAW_FALLBACK;
}

/* ---------------- I2C setup ---------------- */

static esp_err_t lepton_i2c_init(void)
{
    i2c_master_bus_config_t bus_cfg = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = I2C_PORT_NUM,
        .scl_io_num = I2C_SCL_GPIO,
        .sda_io_num = I2C_SDA_GPIO,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };

    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &i2c_bus));

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = LEPTON_I2C_ADDR,
        .scl_speed_hz = I2C_SCL_SPEED_HZ,
    };

    ESP_ERROR_CHECK(i2c_master_bus_add_device(i2c_bus, &dev_cfg, &lepton_i2c_dev));
    return ESP_OK;
}

static esp_err_t lepton_i2c_probe(void)
{
    esp_err_t err = i2c_master_probe(i2c_bus, LEPTON_I2C_ADDR, 100);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Lepton detected on I2C address 0x%02X", LEPTON_I2C_ADDR);
    } else {
        ESP_LOGE(TAG, "Lepton I2C probe failed: %s", esp_err_to_name(err));
    }
    return err;
}

static esp_err_t lepton_cci_write_reg16(uint16_t reg, uint16_t value)
{
    uint8_t tx[4] = {
        (uint8_t)(reg >> 8),
        (uint8_t)(reg & 0xFF),
        (uint8_t)(value >> 8),
        (uint8_t)(value & 0xFF),
    };
    return i2c_master_transmit(lepton_i2c_dev, tx, sizeof(tx), 100);
}

static esp_err_t lepton_cci_read_reg16(uint16_t reg, uint16_t *value)
{
    uint8_t tx[2] = {
        (uint8_t)(reg >> 8),
        (uint8_t)(reg & 0xFF),
    };
    uint8_t rx[2] = {0};

    esp_err_t err = i2c_master_transmit_receive(
        lepton_i2c_dev, tx, sizeof(tx), rx, sizeof(rx), 100
    );
    if (err != ESP_OK) {
        return err;
    }

    *value = ((uint16_t)rx[0] << 8) | rx[1];
    return ESP_OK;
}

static esp_err_t lepton_cci_write_words(uint16_t start_reg, const uint16_t *words, size_t count)
{
    if (count == 0 || count > 16) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t tx[2 + 2 * 16];
    tx[0] = (uint8_t)(start_reg >> 8);
    tx[1] = (uint8_t)(start_reg & 0xFF);

    for (size_t i = 0; i < count; i++) {
        tx[2 + 2*i]     = (uint8_t)(words[i] >> 8);
        tx[2 + 2*i + 1] = (uint8_t)(words[i] & 0xFF);
    }

    return i2c_master_transmit(lepton_i2c_dev, tx, 2 + 2*count, 100);
}

static esp_err_t lepton_cci_read_words(uint16_t start_reg, uint16_t *words, size_t count)
{
    if (count == 0 || count > 16) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t tx[2] = {
        (uint8_t)(start_reg >> 8),
        (uint8_t)(start_reg & 0xFF),
    };
    uint8_t rx[2 * 16] = {0};

    esp_err_t err = i2c_master_transmit_receive(
        lepton_i2c_dev, tx, sizeof(tx), rx, 2 * count, 100
    );
    if (err != ESP_OK) {
        return err;
    }

    for (size_t i = 0; i < count; i++) {
        words[i] = ((uint16_t)rx[2*i] << 8) | rx[2*i + 1];
    }

    return ESP_OK;
}

/* ---------------- SPI setup ---------------- */

static esp_err_t lepton_spi_init(void)
{
    spi_bus_config_t buscfg = {
        .mosi_io_num = LEPTON_SPI_MOSI_GPIO,
        .miso_io_num = LEPTON_SPI_MISO_GPIO,
        .sclk_io_num = LEPTON_SPI_SCK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = LEPTON_PACKET_SIZE,
    };

    spi_device_interface_config_t devcfg = {
        .clock_speed_hz = LEPTON_SPI_CLOCK_HZ,
        .mode = LEPTON_SPI_MODE,
        .spics_io_num = LEPTON_SPI_CS_GPIO,
        .queue_size = 2,
        .cs_ena_pretrans = 0,
        .cs_ena_posttrans = 0,
        .flags = 0,
    };

    ESP_ERROR_CHECK(spi_bus_initialize(LEPTON_SPI_HOST, &buscfg, SPI_DMA_CH_AUTO));
    ESP_ERROR_CHECK(spi_bus_add_device(LEPTON_SPI_HOST, &devcfg, &lepton_spi));

    return ESP_OK;
}

/* ---------------- CCI command helpers ---------------- */

static uint16_t lepton_make_cmd(uint16_t module_id, uint16_t base_plus_type, bool protected_cmd)
{
    return (uint16_t)(module_id | base_plus_type | (protected_cmd ? LEP_CMD_PROTECTION_BIT : 0));
}

static esp_err_t lepton_wait_cmd_idle(int8_t *resp_code, uint32_t timeout_ms)
{
    uint64_t start_us = esp_timer_get_time();

    while ((esp_timer_get_time() - start_us) < (timeout_ms * 1000ULL)) {
        uint16_t status = 0;
        esp_err_t err = lepton_cci_read_reg16(LEP_REG_STATUS, &status);
        if (err != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(2));
            continue;
        }

        bool busy = (status & 0x0001) != 0;
        int8_t rsp = (int8_t)((status >> 8) & 0xFF);

        if (!busy) {
            if (resp_code) {
                *resp_code = rsp;
            }
            return (rsp == 0) ? ESP_OK : ESP_FAIL;
        }

        vTaskDelay(pdMS_TO_TICKS(2));
    }

    return ESP_ERR_TIMEOUT;
}

static esp_err_t lepton_exec_get_enum(uint16_t module, uint16_t base, bool protected_cmd, int32_t *out_val)
{
    int8_t rsp = 0;
    uint16_t words[2] = {0};

    ESP_ERROR_CHECK(lepton_cci_write_reg16(LEP_REG_DATA_LENGTH, 2));
    ESP_ERROR_CHECK(lepton_cci_write_reg16(
        LEP_REG_COMMAND,
        lepton_make_cmd(module, (uint16_t)(base | LEP_CMD_TYPE_GET), protected_cmd)
    ));

    esp_err_t err = lepton_wait_cmd_idle(&rsp, 200);
    if (err != ESP_OK) {
        return err;
    }

    ESP_ERROR_CHECK(lepton_cci_read_words(LEP_REG_DATA0, words, 2));
    *out_val = (int32_t)(((uint32_t)words[1] << 16) | words[0]);
    return ESP_OK;
}

static esp_err_t lepton_exec_set_enum(uint16_t module, uint16_t base, bool protected_cmd, int32_t val)
{
    int8_t rsp = 0;
    uint16_t words[2];

    words[0] = (uint16_t)(val & 0xFFFF);
    words[1] = (uint16_t)(((uint32_t)val >> 16) & 0xFFFF);

    ESP_ERROR_CHECK(lepton_cci_write_words(LEP_REG_DATA0, words, 2));
    ESP_ERROR_CHECK(lepton_cci_write_reg16(LEP_REG_DATA_LENGTH, 2));
    ESP_ERROR_CHECK(lepton_cci_write_reg16(
        LEP_REG_COMMAND,
        lepton_make_cmd(module, (uint16_t)(base | LEP_CMD_TYPE_SET), protected_cmd)
    ));

    return lepton_wait_cmd_idle(&rsp, 300);
}

static esp_err_t lepton_exec_set_words(uint16_t module, uint16_t base, bool protected_cmd,
                                       const uint16_t *words, size_t count)
{
    int8_t rsp = 0;

    ESP_ERROR_CHECK(lepton_cci_write_words(LEP_REG_DATA0, words, count));
    ESP_ERROR_CHECK(lepton_cci_write_reg16(LEP_REG_DATA_LENGTH, (uint16_t)count));
    ESP_ERROR_CHECK(lepton_cci_write_reg16(
        LEP_REG_COMMAND,
        lepton_make_cmd(module, (uint16_t)(base | LEP_CMD_TYPE_SET), protected_cmd)
    ));

    return lepton_wait_cmd_idle(&rsp, 300);
}

static esp_err_t lepton_exec_run0(uint16_t module, uint16_t base, bool protected_cmd)
{
    int8_t rsp = 0;

    ESP_ERROR_CHECK(lepton_cci_write_reg16(LEP_REG_DATA_LENGTH, 0));
    ESP_ERROR_CHECK(lepton_cci_write_reg16(
        LEP_REG_COMMAND,
        lepton_make_cmd(module, (uint16_t)(base | LEP_CMD_TYPE_RUN), protected_cmd)
    ));

    return lepton_wait_cmd_idle(&rsp, 1000);
}

static esp_err_t lepton_get_ffc_status(int32_t *ffc_status)
{
    return lepton_exec_get_enum(LEP_MOD_SYS, LEP_SYS_FFC_STATUS_BASE, false, ffc_status);
}

static esp_err_t lepton_wait_ffc_ready(uint32_t timeout_ms)
{
    uint64_t start_us = esp_timer_get_time();

    while ((esp_timer_get_time() - start_us) < (timeout_ms * 1000ULL)) {
        int32_t s = -999;
        if (lepton_get_ffc_status(&s) == ESP_OK && s == LEP_SYS_STATUS_READY) {
            return ESP_OK;
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }

    return ESP_ERR_TIMEOUT;
}

static esp_err_t lepton_set_flux_linear_defaults(float emissivity, float ambient_c)
{
    lepton_flux_linear_params_t p;

    uint16_t kelvin_x100 = celsius_to_kelvin_x100(ambient_c);

    p.sceneEmissivity = emissivity_to_q13(emissivity);
    p.TBkgK           = kelvin_x100;
    p.tauWindow       = 8192;
    p.TWindowK        = kelvin_x100;
    p.tauAtm          = 8192;
    p.TAtmK           = kelvin_x100;
    p.reflWindow      = 0;
    p.TReflK          = kelvin_x100;

    const uint16_t words[8] = {
        p.sceneEmissivity,
        p.TBkgK,
        p.tauWindow,
        p.TWindowK,
        p.tauAtm,
        p.TAtmK,
        p.reflWindow,
        p.TReflK
    };

    return lepton_exec_set_words(LEP_MOD_RAD, LEP_RAD_FLUX_LINEAR_BASE, true, words, 8);
}

static esp_err_t lepton_apply_high_gain_tlinear_once(void)
{
    int32_t agc = -1;
    int32_t rad = -1;
    int32_t gain = -1;
    int32_t auto_res = -1;
    int32_t tlin_res = -1;
    int32_t tlin_en = -1;

    /*
     * FLIR recommends waiting at least 700 ms after reset/power-up before commands.
     */
    vTaskDelay(pdMS_TO_TICKS(1000));

    ESP_ERROR_CHECK(lepton_exec_set_enum(
        LEP_MOD_AGC, LEP_AGC_ENABLE_BASE, false, LEP_AGC_DISABLE
    ));

    ESP_ERROR_CHECK(lepton_exec_set_enum(
        LEP_MOD_RAD, LEP_RAD_ENABLE_BASE, true, LEP_RAD_ENABLE
    ));

    ESP_ERROR_CHECK(lepton_exec_set_enum(
        LEP_MOD_SYS, LEP_SYS_GAIN_MODE_BASE, false, LEP_SYS_GAIN_MODE_HIGH
    ));

    ESP_ERROR_CHECK(lepton_exec_set_enum(
        LEP_MOD_RAD, LEP_RAD_TLINEAR_AUTO_BASE, true, LEP_RAD_DISABLE
    ));

    ESP_ERROR_CHECK(lepton_exec_set_enum(
        LEP_MOD_RAD, LEP_RAD_TLINEAR_RES_BASE, true, LEP_RAD_RESOLUTION_0_01
    ));

    /*
     * Best-effort scene defaults for human scenes.
     * If this fails, continue; TLinear/high-gain may still work.
     */
    esp_err_t flux_err = lepton_set_flux_linear_defaults(0.98f, 22.0f);
    if (flux_err != ESP_OK) {
        ESP_LOGW(TAG, "Flux-linear defaults not applied: %s", esp_err_to_name(flux_err));
    }

    ESP_ERROR_CHECK(lepton_exec_set_enum(
        LEP_MOD_RAD, LEP_RAD_TLINEAR_ENABLE_BASE, true, LEP_RAD_ENABLE
    ));

    ESP_ERROR_CHECK(lepton_exec_run0(
        LEP_MOD_SYS, LEP_SYS_RUN_FFC_BASE, false
    ));
    ESP_ERROR_CHECK(lepton_wait_ffc_ready(2000));

    ESP_ERROR_CHECK(lepton_exec_get_enum(LEP_MOD_AGC, LEP_AGC_ENABLE_BASE, false, &agc));
    ESP_ERROR_CHECK(lepton_exec_get_enum(LEP_MOD_RAD, LEP_RAD_ENABLE_BASE, true, &rad));
    ESP_ERROR_CHECK(lepton_exec_get_enum(LEP_MOD_SYS, LEP_SYS_GAIN_MODE_BASE, false, &gain));
    ESP_ERROR_CHECK(lepton_exec_get_enum(LEP_MOD_RAD, LEP_RAD_TLINEAR_AUTO_BASE, true, &auto_res));
    ESP_ERROR_CHECK(lepton_exec_get_enum(LEP_MOD_RAD, LEP_RAD_TLINEAR_RES_BASE, true, &tlin_res));
    ESP_ERROR_CHECK(lepton_exec_get_enum(LEP_MOD_RAD, LEP_RAD_TLINEAR_ENABLE_BASE, true, &tlin_en));

    if (agc != LEP_AGC_DISABLE ||
        rad != LEP_RAD_ENABLE ||
        gain != LEP_SYS_GAIN_MODE_HIGH ||
        auto_res != LEP_RAD_DISABLE ||
        tlin_res != LEP_RAD_RESOLUTION_0_01 ||
        tlin_en != LEP_RAD_ENABLE) {
        ESP_LOGE(TAG,
                 "Lepton verify failed: agc=%ld rad=%ld gain=%ld auto=%ld res=%ld tlin=%ld",
                 (long)agc, (long)rad, (long)gain,
                 (long)auto_res, (long)tlin_res, (long)tlin_en);
        return ESP_FAIL;
    }

    g_high_gain_enabled = true;
    g_tlinear_enabled = true;
    g_tlinear_0_01k = true;

    ESP_LOGI(TAG, "Lepton configured: HIGH gain + TLinear 0.01K + AGC off");
    return ESP_OK;
}

static esp_err_t lepton_apply_high_gain_tlinear(void)
{
    esp_err_t err = ESP_FAIL;

    g_high_gain_enabled = false;
    g_tlinear_enabled = false;
    g_tlinear_0_01k = false;

    for (int attempt = 1; attempt <= 3; attempt++) {
        ESP_LOGI(TAG, "Applying high-gain TLinear config, attempt %d", attempt);
        err = lepton_apply_high_gain_tlinear_once();
        if (err == ESP_OK) {
            return ESP_OK;
        }
        ESP_LOGW(TAG, "Lepton config attempt %d failed: %s", attempt, esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    return err;
}

/* ---------------- VoSPI helpers ---------------- */

static esp_err_t lepton_read_packet(uint8_t *packet)
{
    spi_transaction_t t = {
        .flags = 0,
        .length = LEPTON_PACKET_SIZE * 8,
        .rxlength = LEPTON_PACKET_SIZE * 8,
        .tx_buffer = NULL,
        .rx_buffer = packet,
    };

    return spi_device_transmit(lepton_spi, &t);
}

static inline bool lepton_packet_is_discard(const uint8_t *packet)
{
    return ((packet[0] & 0x0F) == 0x0F);
}

static inline uint16_t lepton_packet_line_num(const uint8_t *packet)
{
    uint16_t id = ((uint16_t)packet[0] << 8) | packet[1];
    return id & 0x0FFF;
}

static void lepton_clear_frame(uint16_t *frame)
{
    memset(frame, 0, LEPTON_FRAME_PIXELS * sizeof(uint16_t));
}

static esp_err_t lepton_capture_frame(uint16_t *dest)
{
    uint8_t packet[LEPTON_PACKET_SIZE];
    bool line_seen[LEPTON_FRAME_LINES];
    memset(line_seen, 0, sizeof(line_seen));
    lepton_clear_frame(dest);

    int valid_lines = 0;
    int attempts = 0;
    int discard_packets = 0;

    while (valid_lines < LEPTON_FRAME_LINES && attempts < LEPTON_MAX_RESYNC_PACKETS) {
        attempts++;

        esp_err_t err = lepton_read_packet(packet);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "SPI packet read failed: %s", esp_err_to_name(err));
            return err;
        }

        if (lepton_packet_is_discard(packet)) {
            discard_packets++;
            continue;
        }

        uint16_t line = lepton_packet_line_num(packet);
        if (line >= LEPTON_FRAME_LINES) {
            continue;
        }

        for (int x = 0; x < LEPTON_LINE_PIXELS; x++) {
            int frame_idx = line * LEPTON_LINE_PIXELS + x;
            int payload_idx = LEPTON_PACKET_HEADER + (2 * x);

            uint16_t raw = ((uint16_t)packet[payload_idx] << 8) |
                           ((uint16_t)packet[payload_idx + 1]);

            if (g_tlinear_enabled) {
                dest[frame_idx] = raw;          /* full 16-bit TLinear */
            } else {
                dest[frame_idx] = raw & 0x3FFF; /* raw14 */
            }
        }

        if (!line_seen[line]) {
            line_seen[line] = true;
            valid_lines++;
        }
    }

    if (valid_lines != LEPTON_FRAME_LINES) {
        ESP_LOGW(TAG,
                 "Frame capture incomplete: got %d/%d lines, discard=%d, attempts=%d",
                 valid_lines, LEPTON_FRAME_LINES, discard_packets, attempts);
        return ESP_ERR_TIMEOUT;
    }

    return ESP_OK;
}

/* ---------------- Frame analysis ---------------- */

static thermal_summary_t analyze_frame_stats(const uint16_t *frame)
{
    thermal_summary_t s = {0};

    uint32_t sum_all = 0;
    uint32_t sum_left = 0;
    uint32_t sum_center = 0;
    uint32_t sum_right = 0;

    int count_left = 0;
    int count_center = 0;
    int count_right = 0;

    s.max_raw = 0;
    s.min_raw = 0xFFFF;

    for (int y = 0; y < LEPTON_FRAME_LINES; y++) {
        for (int x = 0; x < LEPTON_LINE_PIXELS; x++) {
            int idx = y * LEPTON_LINE_PIXELS + x;
            uint16_t v = frame[idx];

            sum_all += v;

            if (v > s.max_raw) {
                s.max_raw = v;
                s.max_x = x;
                s.max_y = y;
            }

            if (v < s.min_raw) {
                s.min_raw = v;
                s.min_x = x;
                s.min_y = y;
            }

            if (x < LEPTON_LINE_PIXELS / 3) {
                sum_left += v;
                count_left++;
            } else if (x < (2 * LEPTON_LINE_PIXELS) / 3) {
                sum_center += v;
                count_center++;
            } else {
                sum_right += v;
                count_right++;
            }
        }
    }

    s.timestamp_ms = (uint64_t)(esp_timer_get_time() / 1000ULL);
    s.frame_id = ++g_frame_id;
    s.avg_raw = (float)sum_all / LEPTON_FRAME_PIXELS;
    s.left_avg = (float)sum_left / count_left;
    s.center_avg = (float)sum_center / count_center;
    s.right_avg = (float)sum_right / count_right;

    return s;
}

static void analyze_frame_for_header(
    const uint16_t *frame_buf,
    size_t pixels,
    uint16_t *max_value,
    uint16_t *hot_threshold,
    uint8_t *hot_flag)
{
    uint16_t maxv = 0;

    for (size_t i = 0; i < pixels; i++) {
        if (frame_buf[i] > maxv) {
            maxv = frame_buf[i];
        }
    }

    *max_value = maxv;
    *hot_threshold = get_hot_threshold_counts();
    *hot_flag = (maxv >= *hot_threshold) ? 1 : 0;
}

/* ---------------- Real frame wrapper ---------------- */

static bool get_thermal_frame(uint16_t *frame_buf, size_t pixels)
{
    if (pixels != LEPTON_FRAME_PIXELS) {
        return false;
    }

    esp_err_t err = lepton_capture_frame(frame_buf);
    return (err == ESP_OK);
}

/* ---------------- Wi-Fi SoftAP ---------------- */

static void wifi_init_softap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = { 0 };

    memcpy(wifi_config.ap.ssid, AP_SSID, strlen(AP_SSID));
    memcpy(wifi_config.ap.password, AP_PASS, strlen(AP_PASS));
    wifi_config.ap.ssid_len = strlen(AP_SSID);
    wifi_config.ap.channel = AP_CHANNEL;
    wifi_config.ap.max_connection = AP_MAX_CONN;
    wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "ESP32 SoftAP started");
    ESP_LOGI(TAG, "SSID: %s", AP_SSID);
    ESP_LOGI(TAG, "Password: %s", AP_PASS);
    ESP_LOGI(TAG, "AP IP: 192.168.4.1");
    ESP_LOGI(TAG, "Configured server target: %s:%d", SERVER_IP, SERVER_PORT);
}

/* ---------------- TCP client ---------------- */

static int tcp_connect_to_server(void)
{
    struct sockaddr_in dest_addr;
    memset(&dest_addr, 0, sizeof(dest_addr));

    dest_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(SERVER_PORT);

    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Unable to create socket: errno=%d", errno);
        return -1;
    }

    ESP_LOGI(TAG, "Connecting to %s:%d ...", SERVER_IP, SERVER_PORT);

    if (connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr)) != 0) {
        ESP_LOGE(TAG, "Socket connect failed: errno=%d", errno);
        close(sock);
        return -1;
    }

    ESP_LOGI(TAG, "Connected to server");
    return sock;
}

/* ---------------- Sender task ---------------- */

static void thermal_sender_task(void *pvParameters)
{
    uint16_t *frame_buf = frame_buffer;

    while (1) {
        int sock = tcp_connect_to_server();
        if (sock < 0) {
            vTaskDelay(pdMS_TO_TICKS(3000));
            continue;
        }

        while (1) {
            if (!get_thermal_frame(frame_buf, LEPTON_FRAME_PIXELS)) {
                ESP_LOGE(TAG, "Failed to get thermal frame");
                break;
            }

            thermal_summary_t s = analyze_frame_stats(frame_buf);

            uint16_t max_value = 0;
            uint16_t hot_threshold = 0;
            uint8_t hot_flag = 0;

            analyze_frame_for_header(
                frame_buf,
                LEPTON_FRAME_PIXELS,
                &max_value,
                &hot_threshold,
                &hot_flag
            );

            thermal_header_t header;
            memset(&header, 0, sizeof(header));
            header.magic = 0x4D524854;
            header.width = LEPTON_LINE_PIXELS;
            header.height = LEPTON_FRAME_LINES;
            header.frame_id = s.frame_id;
            header.payload_bytes = LEPTON_FRAME_BYTES;
            header.max_value = max_value;
            header.hot_threshold = hot_threshold;
            header.hot_flag = hot_flag;
            header.mode_flags = 0;

            if (g_high_gain_enabled) {
                header.mode_flags |= MODE_FLAG_HIGH_GAIN;
            }
            if (g_tlinear_enabled) {
                header.mode_flags |= MODE_FLAG_TLINEAR;
            }
            if (g_tlinear_0_01k) {
                header.mode_flags |= MODE_FLAG_TLINEAR_0_01K;
            }

            if (send_all(sock, &header, sizeof(header)) < 0) {
                ESP_LOGE(TAG, "Failed sending header");
                break;
            }

            if (send_all(sock, frame_buf, LEPTON_FRAME_BYTES) < 0) {
                ESP_LOGE(TAG, "Failed sending payload");
                break;
            }

            ESP_LOGI(TAG,
                     "Sent frame %lu | max=%u at (%d,%d) | min=%u | avg=%.2f | hot=%u | flags=0x%02X",
                     (unsigned long)s.frame_id,
                     s.max_raw, s.max_x, s.max_y,
                     s.min_raw,
                     s.avg_raw,
                     header.hot_flag,
                     header.mode_flags);
        }

        shutdown(sock, 0);
        close(sock);
        ESP_LOGW(TAG, "Socket closed, retrying...");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Starting Lepton Wi-Fi thermal streamer");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(lepton_i2c_init());
    ESP_ERROR_CHECK(lepton_spi_init());
    ESP_ERROR_CHECK(lepton_i2c_probe());

    uint16_t regval = 0;
    esp_err_t reg_err = lepton_cci_read_reg16(LEP_REG_STATUS, &regval);
    if (reg_err == ESP_OK) {
        ESP_LOGI(TAG, "CCI status reg = 0x%04X", regval);
    } else {
        ESP_LOGW(TAG, "CCI status read failed: %s", esp_err_to_name(reg_err));
    }

    /*
     * Fail fast if radiometry / TLinear / high gain does not verify.
     * This prevents silently streaming raw counts when you expect temperatures.
     */
    ESP_ERROR_CHECK(lepton_apply_high_gain_tlinear());

    wifi_init_softap();

    xTaskCreate(
        thermal_sender_task,
        "thermal_sender_task",
        16384,
        NULL,
        5,
        NULL
    );
}
