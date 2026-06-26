CC       = gcc
CFLAGS   = -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -std=c11 -O2 -g
LDFLAGS  = -lm

SRCDIR   = src
INCDIR   = include
TESTDIR  = tests
EXAMPLEDIR = examples
BUILDDIR = build

SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
TESTS    = $(wildcard $(TESTDIR)/*.c)
TESTBINS = $(TESTS:$(TESTDIR)/%.c=$(BUILDDIR)/test_%)
EXAMPLES = $(wildcard $(EXAMPLEDIR)/*.c)
EXAMPLEBINS = $(EXAMPLES:$(EXAMPLEDIR)/%.c=$(BUILDDIR)/example_%)

INCLUDES = -I$(INCDIR)

.PHONY: all clean test examples

all: test examples

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Test binaries
$(BUILDDIR)/test_%: $(TESTDIR)/%.c $(OBJS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(OBJS) $(LDFLAGS) -o $@

# Example binaries
$(BUILDDIR)/example_%: $(EXAMPLEDIR)/%.c $(OBJS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(OBJS) $(LDFLAGS) -o $@

# Run all tests
test: $(TESTBINS)
	@echo "=== Running Tests ==="
	@failed=0; \
	for t in $(TESTBINS); do \
		if [ -x "$$t" ]; then \
			echo "--- $$t ---"; \
			./$$t && echo "PASS: $$t" || { echo "FAIL: $$t"; failed=$$((failed+1)); }; \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "=== ALL TESTS PASSED ==="; \
	else \
		echo "=== $$failed TEST(S) FAILED ==="; \
		exit 1; \
	fi

# Build examples
examples: $(EXAMPLEBINS)

# Clean
clean:
	rm -rf $(BUILDDIR)

# Print stats
stats:
	@echo "Headers: $$(find include -name '*.h' | xargs cat | wc -l) lines"
	@echo "Source:  $$(find src -name '*.c' | xargs cat | wc -l) lines"
	@echo "Tests:   $$(find tests -name '*.c' | xargs cat | wc -l) lines"
	@echo "Examples: $$(find examples -name '*.c' | xargs cat | wc -l) lines"
