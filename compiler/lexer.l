%{
#include "y.tab.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* triton = both idioms present: `import triton` AND `@triton.jit` */
int triton_import_seen = 0;
int triton_jit_seen = 0;
%}

%option noyywrap
%option yylineno
%option noinput nounput

DIGIT       [0-9]
LETTER      [a-zA-Z_]
EXPONENT    [eE][+-]?{DIGIT}+

%%

 /* COMMENT */
"#"[^\n]*                       { /* skip comments */ }

 /* KEYWORDS */
"def"                           { return DEF; }
"return"                        { return RETURN; }
"if"                            { return IF; }
"elif"                          { return ELIF; }
"else"                          { return ELSE; }
"for"                           { return FOR; }
"while"                         { return WHILE; }
"in"                            { return IN; }
"and"                           { return AND; }
"or"                            { return OR; }
"not"                           { return NOT; }
"is"                            { return IS; }
"True"                          { return TRUE; }
"False"                         { return FALSE; }
"None"                          { return NONE; }
"import"                        { return IMPORT; }
"from"                          { return FROM; }
"as"                            { return AS; }
"pass"                          { return PASS; }
"raise"                         { return RAISE; }
"assert"                        { return ASSERT; }
"break"                         { return BREAK; }
"continue"                      { return CONTINUE; }
"global"                        { return GLOBAL; }
"nonlocal"                      { return NONLOCAL; }
"del"                           { return DEL; }

 /* TRITON DETECTOR */
"@triton.jit"                   { triton_jit_seen = 1; yyless(1); return AT; }
"import triton"                 { triton_import_seen = 1; yyless(6); return IMPORT; }

 /* LITERALS */
{DIGIT}+\.{DIGIT}*{EXPONENT}?   { return FLOAT; }
\.{DIGIT}+{EXPONENT}?           { return FLOAT; }
{DIGIT}+{EXPONENT}              { return FLOAT; }
{DIGIT}+                        { return INTEGER; }

\"[^\"\n]*\"                    { return STRING; }
'[^'\n]*'                       { return STRING; }

 /* IDENTIFIERS */
{LETTER}({LETTER}|{DIGIT})*     { return NAME; }

 /* OPERATORS */
"+"                             { return PLUS; }
"-"                             { return MINUS; }
"*"                             { return STAR; }
"**"                            { return DOUBLE_STAR; }
"/"                             { return SLASH; }
"//"                            { return DOUBLE_SLASH; }
"%"                             { return PERCENT; }
"@"                             { return AT; }

 /* COMPARISONS */
"<"                             { return LESS; }
"<="                            { return LESS_EQUAL; }
">"                             { return GREATER; }
">="                            { return GREATER_EQUAL; }
"=="                            { return EQUAL_EQUAL; }
"!="                            { return NOT_EQUAL; }

 /* BITWISE */
"&"                             { return AMPERSAND; }
"|"                             { return PIPE; }
"^"                             { return CARET; }
"~"                             { return TILDE; }
"<<"                            { return LEFT_SHIFT; }
">>"                            { return RIGHT_SHIFT; }

 /* ASSIGNMENT */
"="                             { return ASSIGN; }
"+="                            { return PLUS_ASSIGN; }
"-="                            { return MINUS_ASSIGN; }
"*="                            { return STAR_ASSIGN; }
"/="                            { return SLASH_ASSIGN; }
"%="                            { return PERCENT_ASSIGN; }
"&="                            { return AMPERSAND_ASSIGN; }
"|="                            { return PIPE_ASSIGN; }
"^="                            { return CARET_ASSIGN; }
"<<="                           { return LEFT_SHIFT_ASSIGN; }
">>="                           { return RIGHT_SHIFT_ASSIGN; }
"**="                           { return DOUBLE_STAR_ASSIGN; }
"//="                           { return DOUBLE_SLASH_ASSIGN; }

 /* DELIMITERS */
"("                             { return OPEN_PARENTHESIS; }
")"                             { return CLOSE_PARENTHESIS; }
"["                             { return OPEN_BRACKET; }
"]"                             { return CLOSE_BRACKET; }
"{"                             { return OPEN_BRACE; }
"}"                             { return CLOSE_BRACE; }
","                             { return COMMA; }
":"                             { return COLON; }
"."                             { return DOT; }
";"                             { return SEMICOLON; }
"->"                            { return ARROW; }
"..."                           { return ELLIPSIS; }

 /* LAYOUT */
 /* layout-free: newlines are skipped like any other whitespace */
[ \t\r\n]+                      { /* skip all whitespace */ }
<<EOF>>                         { return 0; }

 /* anything else is a lexical error */
.                               { return ERROR; }

%%

#ifdef SCANNER_MAIN
#define TOKEN_LIST(X) \
    X(NAME) X(INTEGER) X(FLOAT) X(STRING) \
    X(DEF) X(RETURN) X(IF) X(ELIF) X(ELSE) X(FOR) X(WHILE) X(IN) \
    X(AND) X(OR) X(NOT) X(IS) X(TRUE) X(FALSE) X(NONE) X(IMPORT) \
    X(FROM) X(AS) X(PASS) X(RAISE) X(ASSERT) X(BREAK) X(CONTINUE) \
    X(GLOBAL) X(NONLOCAL) X(DEL) \
    X(PLUS) X(MINUS) X(STAR) X(DOUBLE_STAR) X(SLASH) X(DOUBLE_SLASH) \
    X(PERCENT) X(AT) X(LESS) X(LESS_EQUAL) X(GREATER) X(GREATER_EQUAL) \
    X(EQUAL_EQUAL) X(NOT_EQUAL) X(AMPERSAND) X(PIPE) X(CARET) X(TILDE) \
    X(LEFT_SHIFT) X(RIGHT_SHIFT) X(ASSIGN) X(PLUS_ASSIGN) X(MINUS_ASSIGN) \
    X(STAR_ASSIGN) X(SLASH_ASSIGN) X(PERCENT_ASSIGN) X(AMPERSAND_ASSIGN) \
    X(PIPE_ASSIGN) X(CARET_ASSIGN) X(LEFT_SHIFT_ASSIGN) X(RIGHT_SHIFT_ASSIGN) \
    X(DOUBLE_STAR_ASSIGN) X(DOUBLE_SLASH_ASSIGN) \
    X(OPEN_PARENTHESIS) X(CLOSE_PARENTHESIS) X(OPEN_BRACKET) X(CLOSE_BRACKET) \
    X(OPEN_BRACE) X(CLOSE_BRACE) X(COMMA) X(COLON) X(DOT) X(SEMICOLON) \
    X(ARROW) X(ELLIPSIS) X(ERROR)

static const char *token_name(int token)
{
    switch (token) {
#define TOKEN_CASE(name) case name: return #name;
    TOKEN_LIST(TOKEN_CASE)
#undef TOKEN_CASE
    default: return "UNKNOWN";
    }
}

#define MAX_SYMBOLS 4096
#define MAX_LEXEME  64

typedef struct {
    int  token;
    char lexeme[MAX_LEXEME];
    int  first_line;
    int  count;
} Symbol;

static Symbol symbols[MAX_SYMBOLS];
static int symbol_count = 0;

static int is_symbol_token(int token)
{
    return token == NAME || token == INTEGER || token == FLOAT || token == STRING;
}

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

static void add_symbol(int token, const char *lexeme, int line)
{
    char key[MAX_LEXEME];
    int i;

    strncpy(key, lexeme, MAX_LEXEME - 1);
    key[MAX_LEXEME - 1] = '\0';
    if (strlen(lexeme) >= MAX_LEXEME)
        strcpy(key + MAX_LEXEME - 4, "...");

    for (i = 0; i < symbol_count; i++) {
        if (symbols[i].token == token && strcmp(symbols[i].lexeme, key) == 0) {
            symbols[i].count++;
            return;
        }
    }
    if (symbol_count < MAX_SYMBOLS) {
        symbols[symbol_count].token = token;
        strcpy(symbols[symbol_count].lexeme, key);
        symbols[symbol_count].first_line = line;
        symbols[symbol_count].count = 1;
        symbol_count++;
    }
}

static void print_symbol_table(void)
{
    int i;

    printf("\n================== SYMBOL TABLE ==================\n");
    printf("%-5s %-9s %-32s %-11s %s\n",
           "ID", "TOKEN", "LEXEME", "FIRST_LINE", "COUNT");
    for (i = 0; i < symbol_count; i++)
        printf("%-5d %-9s %-32s %-11d %d\n",
               i + 1, token_name(symbols[i].token), symbols[i].lexeme,
               symbols[i].first_line, symbols[i].count);
    printf("(%d distinct symbols)\n", symbol_count);
}

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
        if (is_symbol_token(token))
            add_symbol(token, display(yytext), yylineno);
    }
    printf("EOF\n");

    print_symbol_table();

    printf("\ntriton: %s\n",
           (triton_import_seen && triton_jit_seen) ? "true" : "false");

    if (lexical_errors) {
        printf("\n%d lexical error(s) found.\n", lexical_errors);
        return 1;
    }
    printf("\nNo lexical errors.\n");
    return 0;
}

#undef TOKEN_LIST
#endif
