FLEX ?= flex
CC ?= cc
CFLAGS ?= -Wall -Wextra -std=c99 -DTRITON_LEXER_STANDALONE

all: triton_lexer

lex.yy.c: triton_lexer.l triton_tokens.h
	$(FLEX) -o $@ triton_lexer.l

triton_lexer: lex.yy.c triton_tokens.h
	$(CC) $(CFLAGS) -o $@ lex.yy.c

clean:
	rm -f lex.yy.c triton_lexer
