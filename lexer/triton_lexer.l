%option noyywrap noinput nounput nodefault
%x BOL TRI_DQ TRI_SQ

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "triton_tokens.h"

#define YY_DECL static int triton_raw_lex(void)

#define TOKEN_TEXT_MAX 4096
#define PENDING_MAX 512
#define INDENT_MAX 256

typedef struct pending_token {
    int token;
    int line;
    int column;
    char text[32];
} pending_token;

char triton_token_text[TOKEN_TEXT_MAX];
int triton_token_line = 1;
int triton_token_column = 1;

static int scanner_line = 1;
static int scanner_column = 1;
static int indent_stack[INDENT_MAX] = {0};
static int indent_top = 0;
static int nesting_depth = 0;
static int saw_eof = 0;
static int string_start_line = 1;
static int string_start_column = 1;
static pending_token pending[PENDING_MAX];
static int pending_head = 0;
static int pending_tail = 0;

static void set_token_text(const char *text, int length)
{
    int copy_len = length;
    if (copy_len < 0) {
        copy_len = 0;
    }
    if (copy_len >= TOKEN_TEXT_MAX) {
        copy_len = TOKEN_TEXT_MAX - 1;
    }
    memcpy(triton_token_text, text, (size_t)copy_len);
    triton_token_text[copy_len] = '\0';
}

static void set_current_token(int token, int line, int column, const char *text)
{
    (void)token;
    triton_token_line = line;
    triton_token_column = column;
    set_token_text(text, (int)strlen(text));
}

static void update_position(const char *text, int length)
{
    int i;
    for (i = 0; i < length; i++) {
        if (text[i] == '\n') {
            scanner_line++;
            scanner_column = 1;
        } else {
            scanner_column++;
        }
    }
}

#define YY_USER_ACTION do { \
    triton_token_line = scanner_line; \
    triton_token_column = scanner_column; \
    set_token_text(yytext, yyleng); \
    update_position(yytext, yyleng); \
} while (0);

#define RET(token) return (token)

static int open_delimiter(int token)
{
    nesting_depth++;
    return token;
}

static int close_delimiter(int token)
{
    if (nesting_depth > 0) {
        nesting_depth--;
    }
    return token;
}

static int pending_empty(void)
{
    return pending_head == pending_tail;
}

static void enqueue_token(int token, int line, int column, const char *text)
{
    int next_tail = (pending_tail + 1) % PENDING_MAX;
    if (next_tail == pending_head) {
        fprintf(stderr, "triton lexer: pending token queue overflow\n");
        exit(2);
    }

    pending[pending_tail].token = token;
    pending[pending_tail].line = line;
    pending[pending_tail].column = column;
    snprintf(pending[pending_tail].text, sizeof(pending[pending_tail].text), "%s", text);
    pending_tail = next_tail;
}

static int pop_pending(void)
{
    pending_token item;
    if (pending_empty()) {
        return 0;
    }

    item = pending[pending_head];
    pending_head = (pending_head + 1) % PENDING_MAX;
    set_current_token(item.token, item.line, item.column, item.text);
    return item.token;
}

static int indentation_width(const char *text, int length)
{
    int i;
    int width = 0;

    for (i = 0; i < length; i++) {
        if (text[i] == ' ') {
            width++;
        } else if (text[i] == '\t') {
            width += 8 - (width % 8);
        } else if (text[i] == '\f') {
            width = 0;
        }
    }

    return width;
}

static void process_indentation(const char *text, int length)
{
    int width = indentation_width(text, length);
    int line = triton_token_line;

    if (width > indent_stack[indent_top]) {
        if (indent_top + 1 >= INDENT_MAX) {
            enqueue_token(T_ERROR, line, 1, "<INDENT_OVERFLOW>");
            return;
        }
        indent_top++;
        indent_stack[indent_top] = width;
        enqueue_token(T_INDENT, line, 1, "<INDENT>");
        return;
    }

    if (width < indent_stack[indent_top]) {
        while (indent_top > 0 && width < indent_stack[indent_top]) {
            indent_top--;
            enqueue_token(T_DEDENT, line, 1, "<DEDENT>");
        }
        if (width != indent_stack[indent_top]) {
            enqueue_token(T_ERROR, line, 1, "<BAD_DEDENT>");
        }
    }
}
%}

IDENT      [A-Za-z_][A-Za-z0-9_]*
EXP        [eE][+-]?[0-9][0-9_]*
INT        (0[xX][0-9A-Fa-f_]+|0[bB][01_]+|0[oO][0-7_]+|[0-9][0-9_]*)
FLOAT      (([0-9][0-9_]*\.[0-9_]*|[0-9_]*\.[0-9][0-9_]*){EXP}?|[0-9][0-9_]*{EXP})
PREFIX     ([rRuUbBfF]|[rR][fF]|[fF][rR]|[bB][rR]|[rR][bB])?

%%

<BOL>[ \t\f]*\r?\n              { BEGIN(BOL); }
<BOL>[ \t\f]*#[^\r\n]*(\r?\n)?  { BEGIN(BOL); }
<BOL>[ \t\f]+                   { if (nesting_depth == 0) { process_indentation(yytext, yyleng); } BEGIN(INITIAL); if (!pending_empty()) RET(pop_pending()); }
<BOL>.                          { if (nesting_depth == 0) { process_indentation("", 0); } BEGIN(INITIAL); scanner_line = triton_token_line; scanner_column = triton_token_column; yyless(0); if (!pending_empty()) RET(pop_pending()); }

\\[ \t\f]*\r?\n                 { }
[ \t\f]+                         { }
#[^\r\n]*                        { }
\r?\n                            { BEGIN(BOL); if (nesting_depth == 0) { RET(T_NEWLINE); } }

{PREFIX}\"\"\"                   { string_start_line = triton_token_line; string_start_column = triton_token_column; BEGIN(TRI_DQ); }
<TRI_DQ>\"\"\"                   { BEGIN(INITIAL); set_current_token(T_STRING, string_start_line, string_start_column, "<TRIPLE_STRING>"); RET(T_STRING); }
<TRI_DQ>\\.|\n|.                 { }
<TRI_DQ><<EOF>>                  { set_current_token(T_ERROR, string_start_line, string_start_column, "<UNTERMINATED_STRING>"); RET(T_ERROR); }

{PREFIX}\'\'\'                   { string_start_line = triton_token_line; string_start_column = triton_token_column; BEGIN(TRI_SQ); }
<TRI_SQ>\'\'\'                   { BEGIN(INITIAL); set_current_token(T_STRING, string_start_line, string_start_column, "<TRIPLE_STRING>"); RET(T_STRING); }
<TRI_SQ>\\.|\n|.                 { }
<TRI_SQ><<EOF>>                  { set_current_token(T_ERROR, string_start_line, string_start_column, "<UNTERMINATED_STRING>"); RET(T_ERROR); }

{PREFIX}\"([^\\\"\r\n]|\\.)*\"   { RET(T_STRING); }
{PREFIX}\'([^\\\'\r\n]|\\.)*\'   { RET(T_STRING); }
{PREFIX}\"([^\\\"\r\n]|\\.)*     { RET(T_ERROR); }
{PREFIX}\'([^\\\'\r\n]|\\.)*     { RET(T_ERROR); }

"def"                            { RET(T_DEF); }
"return"                         { RET(T_RETURN); }
"if"                             { RET(T_IF); }
"elif"                           { RET(T_ELIF); }
"else"                           { RET(T_ELSE); }
"for"                            { RET(T_FOR); }
"while"                          { RET(T_WHILE); }
"in"                             { RET(T_IN); }
"import"                         { RET(T_IMPORT); }
"from"                           { RET(T_FROM); }
"as"                             { RET(T_AS); }
"try"                            { RET(T_TRY); }
"except"                         { RET(T_EXCEPT); }
"finally"                        { RET(T_FINALLY); }
"pass"                           { RET(T_PASS); }
"break"                          { RET(T_BREAK); }
"continue"                       { RET(T_CONTINUE); }
"class"                          { RET(T_CLASS); }
"with"                           { RET(T_WITH); }
"lambda"                         { RET(T_LAMBDA); }
"yield"                          { RET(T_YIELD); }
"assert"                         { RET(T_ASSERT); }
"raise"                          { RET(T_RAISE); }
"global"                         { RET(T_GLOBAL); }
"nonlocal"                       { RET(T_NONLOCAL); }
"del"                            { RET(T_DEL); }
"async"                          { RET(T_ASYNC); }
"await"                          { RET(T_AWAIT); }
"and"                            { RET(T_AND); }
"or"                             { RET(T_OR); }
"not"                            { RET(T_NOT); }
"is"                             { RET(T_IS); }
"None"                           { RET(T_NONE); }
"True"                           { RET(T_TRUE); }
"False"                          { RET(T_FALSE); }

{FLOAT}[jJ]                      { RET(T_COMPLEX); }
{INT}[jJ]                        { RET(T_COMPLEX); }
{FLOAT}                          { RET(T_FLOAT); }
{INT}                            { RET(T_INTEGER); }
{IDENT}                          { RET(T_IDENTIFIER); }

"..."                            { RET(T_ELLIPSIS); }
"->"                             { RET(T_ARROW); }
":="                             { RET(T_WALRUS); }
"=="                             { RET(T_EQ); }
"!="                             { RET(T_NE); }
"<="                             { RET(T_LE); }
">="                             { RET(T_GE); }
"<<="                            { RET(T_LSHIFT_ASSIGN); }
">>="                            { RET(T_RSHIFT_ASSIGN); }
"**="                            { RET(T_POWER_ASSIGN); }
"//="                            { RET(T_FLOORDIV_ASSIGN); }
"+="                             { RET(T_PLUS_ASSIGN); }
"-="                             { RET(T_MINUS_ASSIGN); }
"*="                             { RET(T_STAR_ASSIGN); }
"/="                             { RET(T_SLASH_ASSIGN); }
"%="                             { RET(T_PERCENT_ASSIGN); }
"&="                             { RET(T_AMP_ASSIGN); }
"|="                             { RET(T_PIPE_ASSIGN); }
"^="                             { RET(T_CARET_ASSIGN); }
"@="                             { RET(T_AT_ASSIGN); }
"<<"                             { RET(T_LSHIFT); }
">>"                             { RET(T_RSHIFT); }
"**"                             { RET(T_POWER); }
"//"                             { RET(T_FLOORDIV); }
"="                              { RET(T_ASSIGN); }
"<"                              { RET(T_LT); }
">"                              { RET(T_GT); }
"+"                              { RET(T_PLUS); }
"-"                              { RET(T_MINUS); }
"*"                              { RET(T_STAR); }
"/"                              { RET(T_SLASH); }
"%"                              { RET(T_PERCENT); }
"&"                              { RET(T_AMP); }
"|"                              { RET(T_PIPE); }
"^"                              { RET(T_CARET); }
"~"                              { RET(T_TILDE); }
"."                              { RET(T_DOT); }
","                              { RET(T_COMMA); }
":"                              { RET(T_COLON); }
";"                              { RET(T_SEMICOLON); }
"@"                              { RET(T_AT); }
"\\"                             { RET(T_BACKSLASH); }
"("                              { RET(open_delimiter(T_LPAREN)); }
")"                              { RET(close_delimiter(T_RPAREN)); }
"["                              { RET(open_delimiter(T_LBRACKET)); }
"]"                              { RET(close_delimiter(T_RBRACKET)); }
"{"                              { RET(open_delimiter(T_LBRACE)); }
"}"                              { RET(close_delimiter(T_RBRACE)); }

.                                { RET(T_ERROR); }

<<EOF>>                          {
                                      if (!saw_eof) {
                                          saw_eof = 1;
                                          while (indent_top > 0) {
                                              indent_top--;
                                              enqueue_token(T_DEDENT, scanner_line, 1, "<DEDENT>");
                                          }
                                      }
                                      if (!pending_empty()) {
                                          RET(pop_pending());
                                      }
                                      return 0;
                                  }

%%

int yylex(void)
{
    int token = pop_pending();
    if (token != 0) {
        return token;
    }
    return triton_raw_lex();
}

const char *triton_token_name(int token)
{
    switch (token) {
    case 0: return "EOF";
    case T_ERROR: return "ERROR";
    case T_NEWLINE: return "NEWLINE";
    case T_INDENT: return "INDENT";
    case T_DEDENT: return "DEDENT";
    case T_IDENTIFIER: return "IDENTIFIER";
    case T_INTEGER: return "INTEGER";
    case T_FLOAT: return "FLOAT";
    case T_COMPLEX: return "COMPLEX";
    case T_STRING: return "STRING";
    case T_DEF: return "DEF";
    case T_RETURN: return "RETURN";
    case T_IF: return "IF";
    case T_ELIF: return "ELIF";
    case T_ELSE: return "ELSE";
    case T_FOR: return "FOR";
    case T_WHILE: return "WHILE";
    case T_IN: return "IN";
    case T_IMPORT: return "IMPORT";
    case T_FROM: return "FROM";
    case T_AS: return "AS";
    case T_TRY: return "TRY";
    case T_EXCEPT: return "EXCEPT";
    case T_FINALLY: return "FINALLY";
    case T_PASS: return "PASS";
    case T_BREAK: return "BREAK";
    case T_CONTINUE: return "CONTINUE";
    case T_CLASS: return "CLASS";
    case T_WITH: return "WITH";
    case T_LAMBDA: return "LAMBDA";
    case T_YIELD: return "YIELD";
    case T_ASSERT: return "ASSERT";
    case T_RAISE: return "RAISE";
    case T_GLOBAL: return "GLOBAL";
    case T_NONLOCAL: return "NONLOCAL";
    case T_DEL: return "DEL";
    case T_ASYNC: return "ASYNC";
    case T_AWAIT: return "AWAIT";
    case T_AND: return "AND";
    case T_OR: return "OR";
    case T_NOT: return "NOT";
    case T_IS: return "IS";
    case T_NONE: return "NONE";
    case T_TRUE: return "TRUE";
    case T_FALSE: return "FALSE";
    case T_ARROW: return "ARROW";
    case T_WALRUS: return "WALRUS";
    case T_ELLIPSIS: return "ELLIPSIS";
    case T_EQ: return "EQ";
    case T_NE: return "NE";
    case T_LE: return "LE";
    case T_GE: return "GE";
    case T_LSHIFT: return "LSHIFT";
    case T_RSHIFT: return "RSHIFT";
    case T_POWER: return "POWER";
    case T_FLOORDIV: return "FLOORDIV";
    case T_PLUS_ASSIGN: return "PLUS_ASSIGN";
    case T_MINUS_ASSIGN: return "MINUS_ASSIGN";
    case T_STAR_ASSIGN: return "STAR_ASSIGN";
    case T_SLASH_ASSIGN: return "SLASH_ASSIGN";
    case T_PERCENT_ASSIGN: return "PERCENT_ASSIGN";
    case T_AMP_ASSIGN: return "AMP_ASSIGN";
    case T_PIPE_ASSIGN: return "PIPE_ASSIGN";
    case T_CARET_ASSIGN: return "CARET_ASSIGN";
    case T_AT_ASSIGN: return "AT_ASSIGN";
    case T_LSHIFT_ASSIGN: return "LSHIFT_ASSIGN";
    case T_RSHIFT_ASSIGN: return "RSHIFT_ASSIGN";
    case T_POWER_ASSIGN: return "POWER_ASSIGN";
    case T_FLOORDIV_ASSIGN: return "FLOORDIV_ASSIGN";
    case T_ASSIGN: return "ASSIGN";
    case T_LT: return "LT";
    case T_GT: return "GT";
    case T_PLUS: return "PLUS";
    case T_MINUS: return "MINUS";
    case T_STAR: return "STAR";
    case T_SLASH: return "SLASH";
    case T_PERCENT: return "PERCENT";
    case T_AMP: return "AMP";
    case T_PIPE: return "PIPE";
    case T_CARET: return "CARET";
    case T_TILDE: return "TILDE";
    case T_DOT: return "DOT";
    case T_COMMA: return "COMMA";
    case T_COLON: return "COLON";
    case T_SEMICOLON: return "SEMICOLON";
    case T_AT: return "AT";
    case T_BACKSLASH: return "BACKSLASH";
    case T_LPAREN: return "LPAREN";
    case T_RPAREN: return "RPAREN";
    case T_LBRACKET: return "LBRACKET";
    case T_RBRACKET: return "RBRACKET";
    case T_LBRACE: return "LBRACE";
    case T_RBRACE: return "RBRACE";
    default: return "UNKNOWN";
    }
}

#ifdef TRITON_LEXER_STANDALONE
static void print_escaped(const char *text)
{
    const unsigned char *p = (const unsigned char *)text;
    while (*p != '\0') {
        switch (*p) {
        case '\n':
            printf("\\n");
            break;
        case '\r':
            printf("\\r");
            break;
        case '\t':
            printf("\\t");
            break;
        default:
            putchar(*p);
            break;
        }
        p++;
    }
}

int main(int argc, char **argv)
{
    int token;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [source.py]\n", argv[0]);
        return 2;
    }

    if (argc == 2) {
        yyin = fopen(argv[1], "r");
        if (yyin == NULL) {
            perror(argv[1]);
            return 1;
        }
    }

    BEGIN(BOL);
    while ((token = yylex()) != 0) {
        printf("%5d:%-4d %-18s ", triton_token_line, triton_token_column, triton_token_name(token));
        print_escaped(triton_token_text);
        putchar('\n');
    }

    if (yyin != NULL && yyin != stdin) {
        fclose(yyin);
    }
    return 0;
}
#endif
