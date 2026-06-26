# Makefile for mini-soft-sensor-maintenance-aging
# Soft Sensor Maintenance & Aging Management

CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -O2 -g
LDFLAGS  = -lm
INCLUDES = -Iinclude

SRC_DIR  = src
TEST_DIR = tests
EXAMPLE_DIR = examples
BUILD_DIR = build

SRCS     = $(wildcard $(SRC_DIR)/*.c)
TESTS    = $(wildcard $(TEST_DIR)/*.c)
EXAMPLES = $(wildcard $(EXAMPLE_DIR)/*.c)

TEST_BINS   = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/test_%,$(TESTS))
EXAMPLE_BINS = $(patsubst $(EXAMPLE_DIR)/%.c,$(BUILD_DIR)/%,$(EXAMPLES))

.PHONY: all test examples clean check

all: test examples

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

LIB_OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_%: $(TEST_DIR)/%.c $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB_OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%: $(EXAMPLE_DIR)/%.c $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB_OBJS) $(LDFLAGS) -o $@

test: $(TEST_BINS)
	@echo "=== Running tests ==="
	@passed=0; failed=0; for t in $(TEST_BINS); do echo "Running: $$t"; if $$t; then passed=$$((passed+1)); else failed=$$((failed+1)); fi; done; echo "=== Results: $$passed passed, $$failed failed ==="; [ $$failed -eq 0 ] || exit 1

examples: $(EXAMPLE_BINS)
	@echo "=== Running examples ==="
	@for e in $(EXAMPLE_BINS); do echo "Running: $$e"; $$e; echo "---"; done

check:
	@echo "=== Safety Review ==="
	@echo "Checking for filler patterns..."
	@grep -rn "_fn[0-9]" include/ src/ tests/ || echo "  No _fnN patterns"
	@grep -rn "_aux[0-9]" include/ src/ tests/ || echo "  No _auxN patterns"
	@grep -rn "_ext[0-9][^a-z]" include/ src/ tests/ || echo "  No _extN patterns"
	@grep -rn "TODO\|FIXME\|stub\|placeholder" include/ src/ tests/ examples/ || echo "  No TODO/FIXME/stub/placeholder"
	@echo "Line counts:"
	@echo -n "  include/ + src/: "
	@find include -name "*.h" -exec cat {} + | wc -l
	@find src -name "*.c" -exec cat {} + | wc -l

clean:
	rm -rf $(BUILD_DIR)
