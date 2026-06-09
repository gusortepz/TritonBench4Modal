BISON ?= bison
FLEX ?= flex
CC ?= cc
CFLAGS ?= -Wall -Wextra -std=c99

LEXER_DIR := ../lexer

all: triton_parser

$(LEXER_DIR)/lex.yy.c: $(LEXER_DIR)/triton_lexer.l $(LEXER_DIR)/triton_tokens.h
	$(FLEX) -o $@ $(LEXER_DIR)/triton_lexer.l

triton_parser.tab.c: triton_parser.y $(LEXER_DIR)/triton_tokens.h
	$(BISON) -d -o $@ triton_parser.y

triton_parser: triton_parser.tab.c $(LEXER_DIR)/lex.yy.c $(LEXER_DIR)/triton_tokens.h
	$(CC) $(CFLAGS) -o $@ triton_parser.tab.c $(LEXER_DIR)/lex.yy.c

clean:
	rm -f triton_parser.tab.c triton_parser.tab.h triton_parser
	rm -f $(LEXER_DIR)/lex.yy.c
