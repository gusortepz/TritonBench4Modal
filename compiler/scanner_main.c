/*
 * scanner_main.c: standalone driver for the Triton scanner (newlexer.l)
 *
 * This driver turns the lex-generated analyzer into the deliverable scanner:
 *   1. prints the SEQUENCE OF TOKENS  (token name, lexeme, line number)
 *   2. builds and prints the SYMBOL TABLE for the tokens that require it
 *      (NAME identifiers and INTEGER / FLOAT / STRING literals)
 *   3. reports LEXICAL ERRORS (invalid symbols) with their line number,
 *      and keeps scanning so every error in the file is reported.
 *
 * Token numbers come from y.tab.h, the same header the parser uses, so the
 * scanner and the parser can never disagree about a token id.
 *
 * Build:  make newscanner          (cc lex.yy.c scanner_main.c)
 * Usage:  ./newscanner kernel.py   or   ./newscanner < kernel.py
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "y.tab.h"

extern int yylex(void);
extern char *yytext;
extern int yylineno;
extern FILE *yyin;

/* ------------------------------------------------ token id -> token name */
static const char *token_name(int t)
{
    switch (t) {
    case NAME: return "NAME";
    case INTEGER: return "INTEGER";
    case FLOAT: return "FLOAT";
    case STRING: return "STRING";
    case DEF: return "DEF";           case RETURN: return "RETURN";
    case IF: return "IF";             case ELIF: return "ELIF";
    case ELSE: return "ELSE";         case FOR: return "FOR";
    case WHILE: return "WHILE";       case IN: return "IN";
    case AND: return "AND";           case OR: return "OR";
    case NOT: return "NOT";           case IS: return "IS";
    case TRUE: return "TRUE";         case FALSE: return "FALSE";
    case NONE: return "NONE";         case IMPORT: return "IMPORT";
    case FROM: return "FROM";         case AS: return "AS";
    case PASS: return "PASS";         case RAISE: return "RAISE";
    case ASSERT: return "ASSERT";     case BREAK: return "BREAK";
    case CONTINUE: return "CONTINUE"; case GLOBAL: return "GLOBAL";
    case NONLOCAL: return "NONLOCAL"; case DEL: return "DEL";
    case PLUS: return "PLUS";         case MINUS: return "MINUS";
    case STAR: return "STAR";         case DOUBLE_STAR: return "DOUBLE_STAR";
    case SLASH: return "SLASH";       case DOUBLE_SLASH: return "DOUBLE_SLASH";
    case PERCENT: return "PERCENT";   case AT: return "AT";
    case LESS: return "LESS";         case LESS_EQUAL: return "LESS_EQUAL";
    case GREATER: return "GREATER";   case GREATER_EQUAL: return "GREATER_EQUAL";
    case EQUAL_EQUAL: return "EQUAL_EQUAL";
    case NOT_EQUAL: return "NOT_EQUAL";
    case AMPERSAND: return "AMPERSAND";
    case PIPE: return "PIPE";         case CARET: return "CARET";
    case TILDE: return "TILDE";       case LEFT_SHIFT: return "LEFT_SHIFT";
    case RIGHT_SHIFT: return "RIGHT_SHIFT";
    case ASSIGN: return "ASSIGN";
    case PLUS_ASSIGN: return "PLUS_ASSIGN";
    case MINUS_ASSIGN: return "MINUS_ASSIGN";
    case STAR_ASSIGN: return "STAR_ASSIGN";
    case SLASH_ASSIGN: return "SLASH_ASSIGN";
    case PERCENT_ASSIGN: return "PERCENT_ASSIGN";
    case AMPERSAND_ASSIGN: return "AMPERSAND_ASSIGN";
    case PIPE_ASSIGN: return "PIPE_ASSIGN";
    case CARET_ASSIGN: return "CARET_ASSIGN";
    case LEFT_SHIFT_ASSIGN: return "LEFT_SHIFT_ASSIGN";
    case RIGHT_SHIFT_ASSIGN: return "RIGHT_SHIFT_ASSIGN";
    case DOUBLE_STAR_ASSIGN: return "DOUBLE_STAR_ASSIGN";
    case DOUBLE_SLASH_ASSIGN: return "DOUBLE_SLASH_ASSIGN";
    case OPEN_PARENTHESIS: return "OPEN_PARENTHESIS";
    case CLOSE_PARENTHESIS: return "CLOSE_PARENTHESIS";
    case OPEN_BRACKET: return "OPEN_BRACKET";
    case CLOSE_BRACKET: return "CLOSE_BRACKET";
    case OPEN_BRACE: return "OPEN_BRACE";
    case CLOSE_BRACE: return "CLOSE_BRACE";
    case COMMA: return "COMMA";       case COLON: return "COLON";
    case DOT: return "DOT";           case SEMICOLON: return "SEMICOLON";
    case ARROW: return "ARROW";       case ELLIPSIS: return "ELLIPSIS";
    case ERROR: return "ERROR";
    default: return "UNKNOWN";
    }
}

/* --------------------------------------------------------- symbol table */
/* One entry per distinct lexeme of a symbol-table token (NAME, INTEGER,  */
/* FLOAT, STRING). Linear search is enough at kernel-file scale; the      */
/* lookup function is the single place to swap in hashing later.          */
#define MAX_SYMBOLS 4096
#define MAX_LEXEME  64        /* long docstrings are truncated for display */

struct symbol {
    int  id;                  /* symbol id, assigned on first appearance  */
    int  token;               /* token id from y.tab.h                    */
    char lexeme[MAX_LEXEME];
    int  first_line;
    int  count;
};

static struct symbol symtab[MAX_SYMBOLS];
static int n_symbols = 0;

static void symtab_insert(int token, const char *lexeme, int line)
{
    char key[MAX_LEXEME];
    int i;

    strncpy(key, lexeme, MAX_LEXEME - 1);
    key[MAX_LEXEME - 1] = '\0';
    if (strlen(lexeme) >= MAX_LEXEME)             /* mark truncation */
        strcpy(key + MAX_LEXEME - 4, "...");

    for (i = 0; i < n_symbols; i++) {
        if (symtab[i].token == token && strcmp(symtab[i].lexeme, key) == 0) {
            symtab[i].count++;
            return;
        }
    }
    if (n_symbols < MAX_SYMBOLS) {
        symtab[n_symbols].id = n_symbols + 1;
        symtab[n_symbols].token = token;
        strcpy(symtab[n_symbols].lexeme, key);
        symtab[n_symbols].first_line = line;
        symtab[n_symbols].count = 1;
        n_symbols++;
    }
}

static void symtab_print(void)
{
    int i;
    printf("\n================== SYMBOL TABLE ==================\n");
    printf("%-5s %-9s %-32s %-11s %s\n",
           "ID", "TOKEN", "LEXEME", "FIRST_LINE", "COUNT");
    for (i = 0; i < n_symbols; i++)
        printf("%-5d %-9s %-32s %-11d %d\n",
               symtab[i].id, token_name(symtab[i].token),
               symtab[i].lexeme, symtab[i].first_line, symtab[i].count);
    printf("(%d distinct symbols)\n", n_symbols);
}

/* ------------------------------------------------------- lexeme display */
/* Multi-line lexemes (docstrings) would break the table layout, so the   */
/* printed form escapes newlines and tabs. Tokens are unaffected.         */
static const char *display(const char *lexeme)
{
    static char buf[2 * MAX_LEXEME];
    int i = 0;
    while (*lexeme && i < (int)sizeof(buf) - 3) {
        if (*lexeme == '\n')      { buf[i++] = '\\'; buf[i++] = 'n'; }
        else if (*lexeme == '\t') { buf[i++] = '\\'; buf[i++] = 't'; }
        else                      { buf[i++] = *lexeme; }
        lexeme++;
    }
    buf[i] = '\0';
    return buf;
}

/* ----------------------------------------------------------------- main */
int main(int argc, char **argv)
{
    int token, lexical_errors = 0;

    if (argc > 1) {
        yyin = fopen(argv[1], "r");
        if (!yyin) {
            perror(argv[1]);
            return 2;
        }
    }

    printf("================= TOKEN SEQUENCE =================\n");
    printf("%-22s %-32s %s\n", "TOKEN", "LEXEME", "LINE");
    while ((token = yylex()) != 0) {
        if (token == ERROR) {
            lexical_errors++;
            printf("%-22s %-32.32s %d   <-- LEXICAL ERROR: invalid symbol\n",
                   "ERROR", display(yytext), yylineno);
            fprintf(stderr, "lexical error at line %d: invalid symbol '%s'\n",
                    yylineno, yytext);
            continue;
        }
        printf("%-22s %-32.32s %d\n", token_name(token), display(yytext), yylineno);
        if (token == NAME || token == INTEGER ||
            token == FLOAT || token == STRING)
            symtab_insert(token, display(yytext), yylineno);
    }
    printf("EOF\n");

    symtab_print();

    if (lexical_errors) {
        printf("\n%d lexical error(s) found.\n", lexical_errors);
        return 1;
    }
    printf("\nNo lexical errors.\n");
    return 0;
}
