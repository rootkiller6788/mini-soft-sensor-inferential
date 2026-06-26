# Makefile for mini-soft-sensor-data-driven-pca
# PCA-based Data-Driven Soft Sensor / Inferential Sensor
#
# Reference: Jolliffe, I.T. "Principal Component Analysis" (2nd ed, 2002)
# Reference: Qin, S.J. "Statistical Process Monitoring" (2003)
# Reference: MacGregor, J.F. & Kourti, T. "SPC of Multivariate Processes" (1995)

CC      := gcc
CFLAGS  := -Wall -Wextra -std=c11 -O2 -g -I./include
LDFLAGS := -lm

SRCDIR  := src
INCDIR  := include
TSTDIR  := tests
EXMDIR  := examples
BNCDIR  := benches
BUILDDIR := build

SRCS := $(SRCDIR)/pca_core.c $(SRCDIR)/pca_decomposition.c $(SRCDIR)/pca_monitoring.c $(SRCDIR)/pca_inferential.c $(SRCDIR)/pca_adaptive.c $(SRCDIR)/pca_kernel.c
OBJS := $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

TEST_SRCS := $(wildcard $(TSTDIR)/test_*.c)
TEST_BINS := $(patsubst $(TSTDIR)/test_%.c,$(BUILDDIR)/test_%,$(TEST_SRCS))

EXM_SRCS := $(wildcard $(EXMDIR)/example_*.c)
EXM_BINS := $(patsubst $(EXMDIR)/example_%.c,$(BUILDDIR)/example_%,$(EXM_SRCS))

BNC_SRCS := $(wildcard $(BNCDIR)/bench_*.c)
BNC_BINS := $(patsubst $(BNCDIR)/bench_%.c,$(BUILDDIR)/bench_%,$(BNC_SRCS))

.PHONY: all clean test examples bench lint check

all: $(OBJS)
	@echo "[OK] All object files compiled."

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "  CC $<"

$(BUILDDIR)/test_%: $(TSTDIR)/test_%.c $(OBJS)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "  LD $@"

$(BUILDDIR)/example_%: $(EXMDIR)/example_%.c $(OBJS)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "  LD $@"

$(BUILDDIR)/bench_%: $(BNCDIR)/bench_%.c $(OBJS)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)
	@echo "  LD $@"

test: $(TEST_BINS)
	@echo "=== Running Tests ==="
	@passed=0; failed=0; \
	for t in $(TEST_BINS); do \
		echo "--- $$t ---"; \
		if ./$$t; then passed=$$((passed+1)); echo "  PASS"; \
		else failed=$$((failed+1)); echo "  FAIL"; fi; \
	done; \
	echo "=== Results: $$passed passed, $$failed failed ==="; \
	test $$failed -eq 0

examples: $(EXM_BINS)
	@echo "=== Running Examples ==="
	@for e in $(EXM_BINS); do echo "--- $$e ---"; ./$$e; done

bench: $(BNC_BINS)
	@echo "=== Running Benchmarks ==="
	@for b in $(BNC_BINS); do echo "--- $$b ---"; ./$$b; done

lint:
	@echo "=== Filler Detection Scan ==="
	@! grep -rn "_fn[0-9]" --include="*.c" --include="*.h" $(SRCDIR) $(INCDIR) 2>/dev/null || (echo "FAIL: _fnN pattern" && false)
	@! grep -rn "_aux[0-9]" --include="*.c" --include="*.h" $(SRCDIR) $(INCDIR) 2>/dev/null || (echo "FAIL: _auxN pattern" && false)
	@! grep -rn "_ext[0-9][^a-z]" --include="*.c" --include="*.h" $(SRCDIR) $(INCDIR) 2>/dev/null || (echo "FAIL: _extN pattern" && false)
	@echo "OK: No filler patterns found."

check: lint
	@echo "=== Line Count Check ==="
	@lines=$$(cat $(INCDIR)/*.h $(SRCDIR)/*.c 2>/dev/null | wc -l); \
	echo "include/ + src/ total lines: $$lines"; \
	if [ $$lines -ge 3000 ]; then echo "PASS: >= 3000 lines"; \
	else echo "FAIL: < 3000 lines ($$lines)"; false; fi

clean:
	rm -rf $(BUILDDIR)

distclean: clean
	rm -f $(TSTDIR)/*.o $(EXMDIR)/*.o $(BNCDIR)/*.o

rebuild: clean all test
