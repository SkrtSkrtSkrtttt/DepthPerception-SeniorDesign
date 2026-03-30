#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"

static const char *TAG = "THERMAL_TEST";

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
#define LEPTON_PACKET_SIZE      164
#define LEPTON_PACKET_HEADER    4
#define LEPTON_LINE_PIXELS      80
#define LEPTON_FRAME_LINES      60
#define LEPTON_FRAME_PIXELS     (LEPTON_LINE_PIXELS * LEPTON_FRAME_LINES)

#define LEPTON_MAX_RESYNC_PACKETS   5000
#define FRAME_PERIOD_MS             500

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

static i2c_master_bus_handle_t i2c_bus = NULL;
static i2c_master_dev_handle_t lepton_i2c_dev = NULL;
static spi_device_handle_t lepton_spi = NULL;

static uint16_t frame_buffer[LEPTON_FRAME_PIXELS];
static uint32_t g_frame_id = 0;

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

/* Optional helper for later expansion */
static esp_err_t lepton_cci_read_reg16(uint16_t reg, uint16_t *value)
{
    uint8_t tx[2] = {
        (uint8_t)((reg >> 8) & 0xFF),
        (uint8_t)(reg & 0xFF)
    };
    uint8_t rx[2] = {0};

    esp_err_t err = i2c_master_transmit_receive(lepton_i2c_dev, tx, sizeof(tx), rx, sizeof(rx), 100);
    if (err != ESP_OK) {
        return err;
    }

    *value = ((uint16_t)rx[0] << 8) | rx[1];
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

            dest[frame_idx] = raw & 0x3FFF;
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

    ESP_LOGI(TAG,
             "Frame capture complete: lines=%d discard=%d attempts=%d",
             valid_lines, discard_packets, attempts);

    return ESP_OK;
}

/* ---------------- Frame analysis ---------------- */

static thermal_summary_t analyze_frame(const uint16_t *frame)
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
    s.min_raw = 0x3FFF;
    s.max_x = 0;
    s.max_y = 0;
    s.min_x = 0;
    s.min_y = 0;

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

/* ---------------- Serial monitor logging ---------------- */

static void print_frame_summary_to_serial(const thermal_summary_t *s)
{
    printf("\n====================================================\n");
    printf("THERMAL FRAME SUMMARY\n");
    printf("timestamp_ms : %" PRIu64 "\n", s->timestamp_ms);
    printf("frame_id     : %lu\n", (unsigned long)s->frame_id);
    printf("max_raw      : %u at (%d, %d)\n", s->max_raw, s->max_x, s->max_y);
    printf("min_raw      : %u at (%d, %d)\n", s->min_raw, s->min_x, s->min_y);
    printf("avg_raw      : %.2f\n", s->avg_raw);
    printf("left_avg     : %.2f\n", s->left_avg);
    printf("center_avg   : %.2f\n", s->center_avg);
    printf("right_avg    : %.2f\n", s->right_avg);
    printf("====================================================\n");
    fflush(stdout);
}

static void print_small_frame_sample(const uint16_t *frame)
{
    printf("Top-left 8x4 sample of raw thermal values:\n");

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 8; x++) {
            int idx = y * LEPTON_LINE_PIXELS + x;
            printf("%5u ", frame[idx]);
        }
        printf("\n");
    }

    printf("\n");
    fflush(stdout);
}

/* ---------------- Main test task ---------------- */

static void lepton_test_task(void *arg)
{
    vTaskDelay(pdMS_TO_TICKS(1000));

    ESP_ERROR_CHECK(lepton_i2c_probe());

    uint16_t regval = 0;
    esp_err_t reg_err = lepton_cci_read_reg16(0x0002, &regval);
    if (reg_err == ESP_OK) {
        ESP_LOGI(TAG, "CCI test read succeeded, reg 0x0002 = 0x%04X", regval);
    } else {
        ESP_LOGW(TAG, "CCI test read failed: %s", esp_err_to_name(reg_err));
    }

    while (1) {
        esp_err_t err = lepton_capture_frame(frame_buffer);
        if (err == ESP_OK) {
            thermal_summary_t summary = analyze_frame(frame_buffer);
            print_frame_summary_to_serial(&summary);
            print_small_frame_sample(frame_buffer);
        } else {
            ESP_LOGW(TAG, "Frame capture failed, retrying...");
        }

        vTaskDelay(pdMS_TO_TICKS(FRAME_PERIOD_MS));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Starting thermal sensor serial-monitor test");

    ESP_ERROR_CHECK(lepton_i2c_init());
    ESP_ERROR_CHECK(lepton_spi_init());

    xTaskCreate(lepton_test_task, "lepton_test_task", 8192, NULL, 5, NULL);
}