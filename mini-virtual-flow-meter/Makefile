# Makefile for mini-virtual-flow-meter
# Part of mini-control-engineering-practice

CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O2 -I./include
LDFLAGS  = -lm

SRC = src/virtual_flow_meter.c src/flow_models.c src/fluid_properties.c       src/vfm_state_estimation.c src/vfm_uncertainty.c src/pipeline_geometry.c       src/vfm_data_reconciliation.c src/vfm_multi_phase.c

OBJ = $(SRC:.c=.o)
TEST_SRC = tests/test_vfm.c
TEST_BIN = test_vfm.exe

EXAMPLES = example_orifice.exe example_pump_vfm.exe example_kalman_vfm.exe            example_data_recon.exe example_multiphase.exe

.PHONY: all
all: test run-test

.PHONY: lib
lib: $(OBJ)
	@echo "[OK] Library objects built"

src/%.o: src/%.c include/*.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_BIN): $(OBJ) $(TEST_SRC)
	$(CC) $(CFLAGS) $(OBJ) $(TEST_SRC) -o $(TEST_BIN) $(LDFLAGS)

.PHONY: test
test: $(TEST_BIN)

.PHONY: run-test
run-test: $(TEST_BIN)
	./$(TEST_BIN)

.PHONY: examples
examples: $(EXAMPLES)

example_orifice.exe: $(OBJ) examples/example_orifice_flow.c
	$(CC) $(CFLAGS) $(OBJ) examples/example_orifice_flow.c -o $@ $(LDFLAGS)

example_pump_vfm.exe: $(OBJ) examples/example_pump_vfm.c
	$(CC) $(CFLAGS) $(OBJ) examples/example_pump_vfm.c -o $@ $(LDFLAGS)

example_kalman_vfm.exe: $(OBJ) examples/example_kalman_vfm.c
	$(CC) $(CFLAGS) $(OBJ) examples/example_kalman_vfm.c -o $@ $(LDFLAGS)

example_data_recon.exe: $(OBJ) examples/example_data_recon.c
	$(CC) $(CFLAGS) $(OBJ) examples/example_data_recon.c -o $@ $(LDFLAGS)

example_multiphase.exe: $(OBJ) examples/example_multiphase.c
	$(CC) $(CFLAGS) $(OBJ) examples/example_multiphase.c -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f src/*.o $(TEST_BIN) $(EXAMPLES)

.PHONY: lines
lines:
	@echo "=== Line Counts ==="
	@echo -n "Headers: "; cat include/*.h | wc -l
	@echo -n "Sources: "; cat src/*.c | wc -l

.PHONY: help
help:
	@echo "mini-virtual-flow-meter Makefile"
	@echo "  make all      - build and run tests"
	@echo "  make test     - build test binary"
	@echo "  make run-test - execute tests"
	@echo "  make examples - build all examples"
	@echo "  make clean    - remove build artifacts"
	@echo "  make lines    - count source lines"
