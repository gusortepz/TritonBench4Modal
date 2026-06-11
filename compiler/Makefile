# Triton-Python recognizer: flex lexer + yacc parser (see grammar.md)
CC = cc

all: newparser newscanner

newparser: y.tab.c lex.yy.c
	$(CC) -o newparser y.tab.c lex.yy.c

# standalone scanner: token sequence + symbol table (lexical phase deliverable)
newscanner: lex.yy.c
	$(CC) -DSCANNER_MAIN -o newscanner lex.yy.c

# yacc -d writes y.tab.h (token numbers, shared with the lexer)
# yacc -v writes y.output (the LALR automaton + conflict report)
y.tab.c y.tab.h: newparser.y
	yacc -d -v newparser.y

lex.yy.c: newlexer.l y.tab.h
	flex newlexer.l

test: newparser newscanner
	mkdir -p results
	./newscanner example_kernel.py > results/tokens.txt
	./newparser example_kernel.py

clean:
	rm -f newparser newscanner y.tab.c y.tab.h y.output lex.yy.c
	rm -rf results

.PHONY: all test clean
