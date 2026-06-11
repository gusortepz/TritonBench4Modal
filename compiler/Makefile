CC = cc

all: parser scanner

parser: y.tab.c lex.yy.c
	$(CC) -o parser y.tab.c lex.yy.c

# standalone scanner
scanner: lex.yy.c
	$(CC) -DSCANNER_MAIN -o scanner lex.yy.c

# yacc -d writes y.tab.h
# yacc -v writes y.output
y.tab.c y.tab.h: parser.y
	yacc -d -v parser.y

lex.yy.c: lexer.l y.tab.h
	flex lexer.l

test: parser scanner
	mkdir -p results
	./scanner example_kernel.py > results/tokens.txt
	./parser example_kernel.py

# full test suite over tests/
suite: parser scanner
	./run_suite.sh

clean:
	rm -f parser scanner y.tab.c y.tab.h y.output lex.yy.c
	rm -rf results

.PHONY: all test suite clean
