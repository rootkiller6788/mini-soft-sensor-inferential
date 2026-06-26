# Makefile for mini-soft-sensor-data-reconciliation
# Builds static library and runs tests + examples.

CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -O2 -g
CFLAGS  += -Iinclude
LDFLAGS  = -lm

SRCS     = $(wildcard src/*.c)
OBJS     = $(SRCS:.c=.o)
LIB      = libdr.a

TEST_SRCS = $(wildcard tests/*.c)
TEST_BINS = $(TEST_SRCS:.c=.test)

EXAMPLE_SRCS = $(wildcard examples/*.c)
EXAMPLE_BINS = $(EXAMPLE_SRCS:.c=.ex)

.PHONY: all clean test examples lib

all: lib test examples

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

tests/%.test: tests/%.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -ldr -lm -o $@

examples/%.ex: examples/%.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -ldr -lm -o $@

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "Running $$t..."; ./$$t || exit 1; done
	@echo "All tests passed."

examples: $(EXAMPLE_BINS)
	@for e in $(EXAMPLE_BINS); do echo "Running $$e..."; ./$$e; done

clean:
	rm -f src/*.o $(LIB) tests/*.test examples/*.ex
