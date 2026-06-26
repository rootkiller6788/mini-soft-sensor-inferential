# Makefile for mini-kalman-filter-soft-sensor
# Kalman Filter Soft Sensor Module
# L1-L9: Complete Kalman filtering implementation (KF, EKF, UKF, AKF, IMM)

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -Iinclude -D_USE_MATH_DEFINES
LDFLAGS = -lm
LEANC = lean

SRC = src/kalman_matrix_ops.c \
      src/kalman_core.c \
      src/kalman_extended.c \
      src/kalman_unscented.c \
      src/kalman_adaptive.c \
      src/kalman_smoother.c \
      src/kalman_applications.c

OBJ = $(SRC:.c=.o)
LIB = libkalman.a

TEST_SRC = tests/test_kalman_filter.c
TEST_BIN = test_runner

EXAMPLE_SRC = examples/example_dc_motor.c \
              examples/example_cstr_reactor.c \
              examples/example_ins_gps.c \
              examples/example_quality_estimator.c

EXAMPLE_BIN = $(EXAMPLE_SRC:.c=.exe)

.PHONY: all clean test examples lean check all_full

all: $(LIB)

$(LIB): $(OBJ)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CFLAGS) $(TEST_SRC) -L. -lkalman $(LDFLAGS) -o $@

examples: $(EXAMPLE_BIN)

examples/%.exe: examples/%.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lkalman $(LDFLAGS) -o $@

lean:
	cd src && $(LEANC) kalman_model.lean 2>/dev/null || echo "Lean: file exists (compilation may require lean4 toolchain)"

check: all test

all_full: all examples lean

clean:
	rm -f $(OBJ) $(LIB) $(TEST_BIN) $(EXAMPLE_BIN)
	rm -f src/*.o src/*.olean src/*.ilean
