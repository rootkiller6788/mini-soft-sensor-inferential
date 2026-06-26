# Makefile for mini-neural-network-soft-sensor
# Neural Network Soft Sensor (Inferential Sensor)

CC       := gcc
CFLAGS   := -Wall -Wextra -std=c11 -O2 -g
LDFLAGS  := -lm
INCLUDES := -Iinclude

BUILD_DIR := build

SRC_FILES := src/nn_soft_sensor_core.c src/nn_training.c src/nn_sensor_validation.c

TEST_FILES := tests/test_nn_core.c tests/test_nn_training.c tests/test_nn_soft_sensor.c

EXAMPLE_FILES := examples/example_distillation_soft_sensor.c examples/example_fermentation_soft_sensor.c examples/example_polymer_soft_sensor.c

SRC_OBJS  := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))
TEST_BINS := $(patsubst tests/%.c,$(BUILD_DIR)/%,$(TEST_FILES))
EXAMPLE_BINS := $(patsubst examples/%.c,$(BUILD_DIR)/%,$(EXAMPLE_FILES))

.PHONY: all
all: test examples

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Test binaries
$(BUILD_DIR)/test_nn_core: tests/test_nn_core.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_nn_training: tests/test_nn_training.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_nn_soft_sensor: tests/test_nn_soft_sensor.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

# Example binaries
$(BUILD_DIR)/example_distillation_soft_sensor: examples/example_distillation_soft_sensor.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/example_fermentation_soft_sensor: examples/example_fermentation_soft_sensor.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/example_polymer_soft_sensor: examples/example_polymer_soft_sensor.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

.PHONY: test
test: $(TEST_BINS)
	@echo "=== Running All Tests ==="
	-$(BUILD_DIR)/test_nn_core
	-$(BUILD_DIR)/test_nn_training
	-$(BUILD_DIR)/test_nn_soft_sensor
	@echo "=== All Tests Done ==="

.PHONY: examples
examples: $(EXAMPLE_BINS)
	@echo "=== All examples built ==="

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: lines
lines:
	@find include src -type f \( -name "*.h" -o -name "*.c" \) -exec cat {} + | wc -l
