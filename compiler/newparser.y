%{
#include <stdio.h>
#include <stdlib.h>

int yylex(void);
void yyerror(const char *msg);

extern int yylineno;
extern char *yytext;
/* set by the lexer; triton requires BOTH import triton and @triton.jit */
extern int triton_import_seen;
extern int triton_jit_seen;
%}

/* literals and names */
%token NAME INTEGER FLOAT STRING

/* keywords */
%token DEF RETURN IF ELIF ELSE FOR WHILE IN AND OR NOT IS
%token TRUE FALSE NONE IMPORT FROM AS PASS RAISE ASSERT
%token BREAK CONTINUE GLOBAL NONLOCAL DEL

/* operators */
%token PLUS MINUS STAR DOUBLE_STAR SLASH DOUBLE_SLASH PERCENT AT
%token LESS LESS_EQUAL GREATER GREATER_EQUAL EQUAL_EQUAL NOT_EQUAL
%token AMPERSAND PIPE CARET TILDE LEFT_SHIFT RIGHT_SHIFT

/* assignment operators */
%token ASSIGN PLUS_ASSIGN MINUS_ASSIGN STAR_ASSIGN SLASH_ASSIGN
%token PERCENT_ASSIGN AMPERSAND_ASSIGN PIPE_ASSIGN CARET_ASSIGN
%token LEFT_SHIFT_ASSIGN RIGHT_SHIFT_ASSIGN DOUBLE_STAR_ASSIGN
%token DOUBLE_SLASH_ASSIGN

/* delimiters */
%token OPEN_PARENTHESIS CLOSE_PARENTHESIS OPEN_BRACKET CLOSE_BRACKET
%token OPEN_BRACE CLOSE_BRACE COMMA COLON DOT SEMICOLON ARROW ELLIPSIS

/* lexical error (any character the lexer cannot match) */
%token ERROR

%start program

%%

/* =================================================================
 * 1.1  Program
 * ================================================================= */

program
    : element program
    | /* ε */
    ;

element
    : header
    | stmt
    ;

/* =================================================================
 * 1.2  Headers:
 * ================================================================= */

header
    : IF test COLON
    | ELIF test COLON
    | ELSE COLON
    | WHILE test COLON
    | FOR exprlist IN testlist COLON
    | funcheader
    | decorator
    ;

funcheader
    : DEF NAME parameters ret_annot COLON
    ;

ret_annot
    : ARROW test
    | /* ε */
    ;

decorator
    : AT dotted_name deco_args
    ;

deco_args
    : OPEN_PARENTHESIS call_args CLOSE_PARENTHESIS
    | /* ε */
    ;

/* =================================================================
 * 1.3  Parameters
 * ================================================================= */

parameters
    : OPEN_PARENTHESIS params CLOSE_PARENTHESIS
    ;

params
    : paramlist
    | /* ε */
    ;

paramlist
    : param paramlist_tail
    ;

paramlist_tail
    : COMMA param_after_comma
    | /* ε */
    ;

param_after_comma
    : param paramlist_tail
    | /* ε */                          /* trailing comma: (a, b,) */
    ;

param
    : NAME param_tail
    ;

param_tail
    : COLON test param_default             /* BLOCK: tl.constexpr [= 128] */
    | ASSIGN test                          /* eps=1e-6                    */
    | /* ε */
    ;

param_default
    : ASSIGN test
    | /* ε */
    ;

/* =================================================================
 * 2  Statements
 * ================================================================= */

stmt
    : testlist assign_tail                 /* expression statement / assignment */
    | RETURN return_tail
    | PASS
    | BREAK
    | CONTINUE
    | import_stmt
    | ASSERT test assert_tail
    | RAISE raise_tail
    | GLOBAL name_list
    | NONLOCAL name_list
    | DEL exprlist
    ;

/* ----- 2.1 expression statements and assignments ----- */

assign_tail
    : ASSIGN testlist eq_chain_tail
    | augassign testlist
    | /* ε */
    ;

eq_chain_tail
    : ASSIGN testlist eq_chain_tail        /* a = b = c */
    | /* ε */
    ;

augassign
    : PLUS_ASSIGN  | MINUS_ASSIGN | STAR_ASSIGN  | SLASH_ASSIGN
    | DOUBLE_SLASH_ASSIGN | PERCENT_ASSIGN | DOUBLE_STAR_ASSIGN
    | AMPERSAND_ASSIGN | PIPE_ASSIGN | CARET_ASSIGN
    | LEFT_SHIFT_ASSIGN | RIGHT_SHIFT_ASSIGN
    ;

/* ----- 2.2 control-flow one-liners ----- */

return_tail
    : testlist
    | /* ε */
    ;

assert_tail
    : COMMA test
    | /* ε */
    ;

raise_tail
    : test
    | /* ε */
    ;

name_list
    : NAME name_list_tail
    ;

name_list_tail
    : COMMA NAME name_list_tail
    | /* ε */
    ;

/* ----- 2.3 imports ----- */

import_stmt
    : IMPORT dotted_as_names
    | FROM dotted_name IMPORT import_target
    ;

import_target
    : STAR
    | name_as_names
    ;

dotted_as_names
    : dotted_as_name dotted_as_names_tail
    ;

dotted_as_names_tail
    : COMMA dotted_as_name dotted_as_names_tail
    | /* ε */
    ;

dotted_as_name
    : dotted_name as_opt
    ;

name_as_names
    : name_as_name name_as_names_tail
    ;

name_as_names_tail
    : COMMA name_as_name name_as_names_tail
    | /* ε */
    ;

name_as_name
    : NAME as_opt
    ;

as_opt
    : AS NAME
    | /* ε */
    ;

dotted_name
    : NAME dotted_name_tail
    ;

dotted_name_tail
    : DOT NAME dotted_name_tail
    | /* ε */
    ;

/* =================================================================
 * 3  Expressions: the precedence ladder (loosest -> tightest)
 * ================================================================= */

/* `test` = full expression entry point (conditional deferred) */
test
    : or_test
    ;

/* ----- 3.1 expression lists (tuples without parentheses) ----- */

testlist
    : test testlist_tail
    ;

testlist_tail
    : COMMA testlist_after_comma
    | /* ε */
    ;

testlist_after_comma
    : test testlist_tail
    | /* ε */                          /* trailing comma: (x,) */
    ;

/* comparison-free twin, used by `for` targets and `del` (no IN inside) */
exprlist
    : expr exprlist_tail
    ;

exprlist_tail
    : COMMA exprlist_after_comma
    | /* ε */
    ;

exprlist_after_comma
    : expr exprlist_tail
    | /* ε */
    ;

/* ----- 3.2 the ladder ----- */

or_test
    : and_test or_tail
    ;

or_tail
    : OR and_test or_tail
    | /* ε */
    ;

and_test
    : not_test and_tail
    ;

and_tail
    : AND not_test and_tail
    | /* ε */
    ;

not_test
    : NOT not_test
    | comparison
    ;

comparison
    : expr comp_tail
    ;

comp_tail
    : comp_op expr comp_tail               /* chains: 0 <= i < n */
    | /* ε */
    ;

comp_op
    : LESS | LESS_EQUAL | GREATER | GREATER_EQUAL
    | EQUAL_EQUAL | NOT_EQUAL
    | IN
    | NOT IN
    | IS is_not_opt
    ;

is_not_opt
    : NOT
    | /* ε */
    ;

expr
    : xor_expr bitor_tail
    ;

bitor_tail
    : PIPE xor_expr bitor_tail
    | /* ε */
    ;

xor_expr
    : and_expr xor_tail
    ;

xor_tail
    : CARET and_expr xor_tail
    | /* ε */
    ;

and_expr
    : shift_expr bitand_tail
    ;

bitand_tail
    : AMPERSAND shift_expr bitand_tail
    | /* ε */
    ;

shift_expr
    : arith_expr shift_tail
    ;

shift_tail
    : LEFT_SHIFT arith_expr shift_tail
    | RIGHT_SHIFT arith_expr shift_tail
    | /* ε */
    ;

arith_expr
    : term arith_tail
    ;

arith_tail
    : PLUS term arith_tail
    | MINUS term arith_tail
    | /* ε */
    ;

term
    : factor term_tail
    ;

term_tail
    : STAR factor term_tail
    | SLASH factor term_tail
    | DOUBLE_SLASH factor term_tail
    | PERCENT factor term_tail
    | AT factor term_tail                  /* matmul: p @ v */
    | /* ε */
    ;

factor
    : PLUS factor
    | MINUS factor
    | TILDE factor
    | power
    ;

power
    : atom_expr power_tail
    ;

power_tail
    : DOUBLE_STAR factor                   /* right-assoc; -x**2, 2**-1 ok */
    | /* ε */
    ;

/* ----- 3.3 atoms and trailers ----- */

atom_expr
    : atom trailer_list
    ;

trailer_list
    : trailer trailer_list                 /* tl.load(...)  k[grid](...) */
    | /* ε */
    ;

trailer
    : OPEN_PARENTHESIS call_args CLOSE_PARENTHESIS
    | OPEN_BRACKET subscriptlist CLOSE_BRACKET
    | DOT NAME
    ;

call_args
    : arglist
    | /* ε */
    ;

atom
    : NAME
    | INTEGER
    | FLOAT
    | strings
    | TRUE
    | FALSE
    | NONE
    | ELLIPSIS
    | OPEN_PARENTHESIS paren_body CLOSE_PARENTHESIS
    | OPEN_BRACKET list_body CLOSE_BRACKET
    | OPEN_BRACE dict_body CLOSE_BRACE
    ;

paren_body
    : testlist
    | /* ε */
    ;

list_body
    : testlist
    | /* ε */
    ;

dict_body
    : dict_items
    | /* ε */
    ;

strings
    : STRING strings_tail                  /* implicit concat: "a" "b" */
    ;

strings_tail
    : STRING strings_tail
    | /* ε */
    ;

dict_items
    : dict_item dict_items_tail
    ;

dict_items_tail
    : COMMA dict_items_after_comma
    | /* ε */
    ;

dict_items_after_comma
    : dict_item dict_items_tail
    | /* ε */                          /* trailing comma in {...} */
    ;

dict_item
    : test COLON test
    ;

/* ----- 3.4 call arguments ----- */

arglist
    : argument arglist_tail
    ;

arglist_tail
    : COMMA arglist_after_comma
    | /* ε */
    ;

arglist_after_comma
    : argument arglist_tail
    | /* ε */                          /* trailing comma in calls */
    ;

argument
    : test kwarg_tail
    ;

kwarg_tail
    : ASSIGN test                          /* mask=mask (LHS: bare NAME,  */
    | /* ε */                          /*  checked semantically)      */
    ;

/* ----- 3.5 subscripts and slices ----- */

subscriptlist
    : subscript subs_tail
    ;

subs_tail
    : COMMA subs_after_comma
    | /* ε */
    ;

subs_after_comma
    : subscript subs_tail
    | /* ε */
    ;

subscript
    : test subscript_tail                  /* x[i]  /  x[a: ...]  */
    | COLON slice_upper                    /* x[: ...]            */
    ;

subscript_tail
    : COLON slice_upper
    | /* ε */
    ;

slice_upper
    : test slice_step_opt                  /* x[a:b ...] */
    | COLON step_opt                       /* x[a::c], x[::c] */
    | /* ε */                          /* x[a:], x[:] */
    ;

slice_step_opt
    : COLON step_opt
    | /* ε */
    ;

step_opt
    : test
    | /* ε */
    ;

%%

void yyerror(const char *msg)
{
    fprintf(stderr, "%s at line %d near '%s'\n", msg, yylineno, yytext);
}

extern FILE *yyin;

int main(int argc, char **argv)
{
    if (argc > 1) {
        yyin = fopen(argv[1], "r");
        if (!yyin) {
            perror(argv[1]);
            return 2;
        }
    }
    int ok = (yyparse() == 0);   /* lexing happens in here: flags valid after */
    printf("triton: %s  code: %s\n",
           (triton_import_seen && triton_jit_seen) ? "true" : "false",
           ok ? "accept" : "reject");
    return ok ? 0 : 1;
}
