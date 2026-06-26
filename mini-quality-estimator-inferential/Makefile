# Makefile for mini-quality-estimator-inferential
# Inferential Quality Estimation

CC       := gcc
CFLAGS   := -Wall -Wextra -std=c11 -O2 -g
LDFLAGS  := -lm
INCLUDES := -Iinclude

BUILD_DIR := build

SRC_FILES := src/quality_estimator_core.c src/inferential_model.c src/kalman_quality.c src/bias_correction.c src/multi_rate_fusion.c src/quality_recursive_ls.c
TEST_FILES := tests/test_quality_estimator.c
EXAMPLE_FILES := examples/example_distillation_quality.c examples/example_polymer_melt_index.c examples/example_emissions_monitoring.c
BENCH_FILES := benches/bench_quality_performance.c
DEMO_FILES := demos/demo_quality_trend.c

SRC_OBJS := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))
TEST_BINS := $(patsubst tests/%.c,$(BUILD_DIR)/%,$(TEST_FILES))
EXAMPLE_BINS := $(patsubst examples/%.c,$(BUILD_DIR)/%,$(EXAMPLE_FILES))
BENCH_BINS := $(patsubst benches/%.c,$(BUILD_DIR)/%,$(BENCH_FILES))
DEMO_BINS := $(patsubst demos/%.c,$(BUILD_DIR)/%,$(DEMO_FILES))

.PHONY: all
all: test examples benches demos

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_quality_estimator: tests/test_quality_estimator.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/example_distillation_quality: examples/example_distillation_quality.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/example_polymer_melt_index: examples/example_polymer_melt_index.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/example_emissions_monitoring: examples/example_emissions_monitoring.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

.PHONY: test
test: $(TEST_BINS)
	@echo "=== Running All Tests ==="
	-$(BUILD_DIR)/test_quality_estimator
	@echo "=== All Tests Done ==="

$(BUILD_DIR)/bench_quality_performance: benches/bench_quality_performance.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/demo_quality_trend: demos/demo_quality_trend.c $(SRC_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

.PHONY: examples
examples: $(EXAMPLE_BINS)
	@echo "=== All examples built ==="

.PHONY: benches
benches: $(BENCH_BINS)
	@echo "=== All benches built ==="

.PHONY: demos
demos: $(DEMO_BINS)
	@echo "=== All demos built ==="

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: lines
lines:
	@echo "include/ + src/ lines:"
	@find include src -type f \( -name "*.h" -o -name "*.c" -o -name "*.lean" \) -exec cat {} + | wc -l
