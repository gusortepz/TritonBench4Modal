%{
#include <stdio.h>
#include <stdlib.h>

extern FILE *yyin;
extern char triton_token_text[];
extern int triton_token_line;
extern int triton_token_column;

static int parser_error_count = 0;
int yylex(void);

static void note_error_token(void);
void yyerror(const char *message);
%}

%token T_ERROR 258
%token T_NEWLINE
%token T_INDENT
%token T_DEDENT
%token T_IDENTIFIER
%token T_INTEGER
%token T_FLOAT
%token T_COMPLEX
%token T_STRING
%token T_DEF
%token T_RETURN
%token T_IF
%token T_ELIF
%token T_ELSE
%token T_FOR
%token T_WHILE
%token T_IN
%token T_IMPORT
%token T_FROM
%token T_AS
%token T_TRY
%token T_EXCEPT
%token T_FINALLY
%token T_PASS
%token T_BREAK
%token T_CONTINUE
%token T_CLASS
%token T_WITH
%token T_LAMBDA
%token T_YIELD
%token T_ASSERT
%token T_RAISE
%token T_GLOBAL
%token T_NONLOCAL
%token T_DEL
%token T_ASYNC
%token T_AWAIT
%token T_AND
%token T_OR
%token T_NOT
%token T_IS
%token T_NONE
%token T_TRUE
%token T_FALSE
%token T_ARROW
%token T_WALRUS
%token T_ELLIPSIS
%token T_EQ
%token T_NE
%token T_LE
%token T_GE
%token T_LSHIFT
%token T_RSHIFT
%token T_POWER
%token T_FLOORDIV
%token T_PLUS_ASSIGN
%token T_MINUS_ASSIGN
%token T_STAR_ASSIGN
%token T_SLASH_ASSIGN
%token T_PERCENT_ASSIGN
%token T_AMP_ASSIGN
%token T_PIPE_ASSIGN
%token T_CARET_ASSIGN
%token T_AT_ASSIGN
%token T_LSHIFT_ASSIGN
%token T_RSHIFT_ASSIGN
%token T_POWER_ASSIGN
%token T_FLOORDIV_ASSIGN
%token T_ASSIGN
%token T_LT
%token T_GT
%token T_PLUS
%token T_MINUS
%token T_STAR
%token T_SLASH
%token T_PERCENT
%token T_AMP
%token T_PIPE
%token T_CARET
%token T_TILDE
%token T_DOT
%token T_COMMA
%token T_COLON
%token T_SEMICOLON
%token T_AT
%token T_BACKSLASH
%token T_LPAREN
%token T_RPAREN
%token T_LBRACKET
%token T_RBRACKET
%token T_LBRACE
%token T_RBRACE

%start module

%%

module
    : statement_list final_line_opt
    ;

statement_list
    : /* empty */
    | statement_list statement
    ;

final_line_opt
    : /* empty */
    | simple_line
    ;

statement
    : T_NEWLINE
    | simple_line T_NEWLINE
    | block_header T_NEWLINE T_INDENT statement_list T_DEDENT
    ;

simple_line
    : simple_start
    | simple_start line_tail
    ;

line_tail
    : line_item
    | line_tail line_item
    ;

line_item
    : non_delimiter_token
    | paren_group
    | bracket_group
    | brace_group
    ;

line_item_no_colon
    : non_delimiter_token_no_colon
    | paren_group
    | bracket_group
    | brace_group
    ;

group_items
    : /* empty */
    | group_items line_item
    ;

paren_group
    : T_LPAREN group_items T_RPAREN
    ;

bracket_group
    : T_LBRACKET group_items T_RBRACKET
    ;

brace_group
    : T_LBRACE group_items T_RBRACE
    ;

block_header
    : block_start T_COLON
    | block_start block_header_tail T_COLON
    ;

block_header_tail
    : line_item_no_colon
    | block_header_tail line_item_no_colon
    ;

block_start
    : T_DEF
    | T_IF
    | T_ELIF
    | T_ELSE
    | T_FOR
    | T_WHILE
    | T_TRY
    | T_EXCEPT
    | T_FINALLY
    | T_WITH
    | T_CLASS
    | T_ASYNC
    ;

simple_start
    : T_IDENTIFIER
    | T_INTEGER
    | T_FLOAT
    | T_COMPLEX
    | T_STRING
    | T_RETURN
    | T_IN
    | T_IMPORT
    | T_FROM
    | T_AS
    | T_PASS
    | T_BREAK
    | T_CONTINUE
    | T_LAMBDA
    | T_YIELD
    | T_ASSERT
    | T_RAISE
    | T_GLOBAL
    | T_NONLOCAL
    | T_DEL
    | T_AWAIT
    | T_AND
    | T_OR
    | T_NOT
    | T_IS
    | T_NONE
    | T_TRUE
    | T_FALSE
    | T_ARROW
    | T_WALRUS
    | T_ELLIPSIS
    | T_EQ
    | T_NE
    | T_LE
    | T_GE
    | T_LSHIFT
    | T_RSHIFT
    | T_POWER
    | T_FLOORDIV
    | T_PLUS_ASSIGN
    | T_MINUS_ASSIGN
    | T_STAR_ASSIGN
    | T_SLASH_ASSIGN
    | T_PERCENT_ASSIGN
    | T_AMP_ASSIGN
    | T_PIPE_ASSIGN
    | T_CARET_ASSIGN
    | T_AT_ASSIGN
    | T_LSHIFT_ASSIGN
    | T_RSHIFT_ASSIGN
    | T_POWER_ASSIGN
    | T_FLOORDIV_ASSIGN
    | T_ASSIGN
    | T_LT
    | T_GT
    | T_PLUS
    | T_MINUS
    | T_STAR
    | T_SLASH
    | T_PERCENT
    | T_AMP
    | T_PIPE
    | T_CARET
    | T_TILDE
    | T_DOT
    | T_COMMA
    | T_COLON
    | T_SEMICOLON
    | T_AT
    | T_BACKSLASH
    | paren_group
    | bracket_group
    | brace_group
    | T_ERROR
        {
            note_error_token();
        }
    ;

non_delimiter_token
    : T_IDENTIFIER
    | T_INTEGER
    | T_FLOAT
    | T_COMPLEX
    | T_STRING
    | T_DEF
    | T_RETURN
    | T_IF
    | T_ELIF
    | T_ELSE
    | T_FOR
    | T_WHILE
    | T_IN
    | T_IMPORT
    | T_FROM
    | T_AS
    | T_TRY
    | T_EXCEPT
    | T_FINALLY
    | T_PASS
    | T_BREAK
    | T_CONTINUE
    | T_CLASS
    | T_WITH
    | T_LAMBDA
    | T_YIELD
    | T_ASSERT
    | T_RAISE
    | T_GLOBAL
    | T_NONLOCAL
    | T_DEL
    | T_ASYNC
    | T_AWAIT
    | T_AND
    | T_OR
    | T_NOT
    | T_IS
    | T_NONE
    | T_TRUE
    | T_FALSE
    | T_ARROW
    | T_WALRUS
    | T_ELLIPSIS
    | T_EQ
    | T_NE
    | T_LE
    | T_GE
    | T_LSHIFT
    | T_RSHIFT
    | T_POWER
    | T_FLOORDIV
    | T_PLUS_ASSIGN
    | T_MINUS_ASSIGN
    | T_STAR_ASSIGN
    | T_SLASH_ASSIGN
    | T_PERCENT_ASSIGN
    | T_AMP_ASSIGN
    | T_PIPE_ASSIGN
    | T_CARET_ASSIGN
    | T_AT_ASSIGN
    | T_LSHIFT_ASSIGN
    | T_RSHIFT_ASSIGN
    | T_POWER_ASSIGN
    | T_FLOORDIV_ASSIGN
    | T_ASSIGN
    | T_LT
    | T_GT
    | T_PLUS
    | T_MINUS
    | T_STAR
    | T_SLASH
    | T_PERCENT
    | T_AMP
    | T_PIPE
    | T_CARET
    | T_TILDE
    | T_DOT
    | T_COMMA
    | T_COLON
    | T_SEMICOLON
    | T_AT
    | T_BACKSLASH
    | T_ERROR
        {
            note_error_token();
        }
    ;

non_delimiter_token_no_colon
    : T_IDENTIFIER
    | T_INTEGER
    | T_FLOAT
    | T_COMPLEX
    | T_STRING
    | T_DEF
    | T_RETURN
    | T_IF
    | T_ELIF
    | T_ELSE
    | T_FOR
    | T_WHILE
    | T_IN
    | T_IMPORT
    | T_FROM
    | T_AS
    | T_TRY
    | T_EXCEPT
    | T_FINALLY
    | T_PASS
    | T_BREAK
    | T_CONTINUE
    | T_CLASS
    | T_WITH
    | T_LAMBDA
    | T_YIELD
    | T_ASSERT
    | T_RAISE
    | T_GLOBAL
    | T_NONLOCAL
    | T_DEL
    | T_ASYNC
    | T_AWAIT
    | T_AND
    | T_OR
    | T_NOT
    | T_IS
    | T_NONE
    | T_TRUE
    | T_FALSE
    | T_ARROW
    | T_WALRUS
    | T_ELLIPSIS
    | T_EQ
    | T_NE
    | T_LE
    | T_GE
    | T_LSHIFT
    | T_RSHIFT
    | T_POWER
    | T_FLOORDIV
    | T_PLUS_ASSIGN
    | T_MINUS_ASSIGN
    | T_STAR_ASSIGN
    | T_SLASH_ASSIGN
    | T_PERCENT_ASSIGN
    | T_AMP_ASSIGN
    | T_PIPE_ASSIGN
    | T_CARET_ASSIGN
    | T_AT_ASSIGN
    | T_LSHIFT_ASSIGN
    | T_RSHIFT_ASSIGN
    | T_POWER_ASSIGN
    | T_FLOORDIV_ASSIGN
    | T_ASSIGN
    | T_LT
    | T_GT
    | T_PLUS
    | T_MINUS
    | T_STAR
    | T_SLASH
    | T_PERCENT
    | T_AMP
    | T_PIPE
    | T_CARET
    | T_TILDE
    | T_DOT
    | T_COMMA
    | T_SEMICOLON
    | T_AT
    | T_BACKSLASH
    | T_ERROR
        {
            note_error_token();
        }
    ;

%%

static void note_error_token(void)
{
    parser_error_count++;
    fprintf(
        stderr,
        "lexer error at %d:%d near `%s`\n",
        triton_token_line,
        triton_token_column,
        triton_token_text
    );
}

void yyerror(const char *message)
{
    fprintf(
        stderr,
        "%s at %d:%d near `%s`\n",
        message,
        triton_token_line,
        triton_token_column,
        triton_token_text
    );
}

int main(int argc, char **argv)
{
    int parse_status;

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

    parse_status = yyparse();

    if (yyin != NULL && yyin != stdin) {
        fclose(yyin);
    }

    if (parse_status != 0 || parser_error_count != 0) {
        return 1;
    }

    return 0;
}
