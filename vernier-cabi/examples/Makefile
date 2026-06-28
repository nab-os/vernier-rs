LIB_DIR  := ../../target/release
INC_DIR  := ../include
CFLAGS   := -std=c11 -O2 -Wall -Wextra -I$(INC_DIR)
LDFLAGS  := -L$(LIB_DIR) -lvernier_cabi -Wl,-rpath,$(abspath $(LIB_DIR)) -lm

.PHONY: all clean run

all: detect

detect: detect.c $(INC_DIR)/vernier.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: detect
	./detect

clean:
	rm -f detect
