# Triton Kernel Starter CFG

This grammar is a first target for a Python-hosted Triton syntactic analyzer.
It is deliberately smaller than full Python, but broad enough to parse imports,
decorators, function definitions, assignments, calls, indexing, attributes,
kernel launches, and common control flow in generated TritonBench kernels.

The grammar is written top-down for readability. Yacc/Bison can encode the same
language bottom-up by translating `*`, `+`, and `?` into recursive productions.

## Terminals

Terminals come from `lexer/triton_tokens.h`. Names such as `IDENTIFIER`,
`STRING`, `INDENT`, and `LPAREN` refer to those token names without the `T_`
prefix.

Every lexer token must be mentioned in this grammar and in
`parser/triton_parser.y`. Run `python3 grammar/check_token_coverage.py` after
editing lexer, CFG, or parser files. `ERROR` and `BACKSLASH` are included
through `recovery_stmt` as diagnostic tokens, not as valid Triton/Python
constructs we expect generated kernels to use. A backslash followed by a
physical newline is consumed by the lexer as explicit line joining, so it does
not reach the parser as `BACKSLASH`.

## Module

```bnf
module          ::= statement_list EOF
statement_list  ::= statement statement_list
                  | empty
statement       ::= simple_stmt NEWLINE
                  | compound_stmt
                  | NEWLINE
simple_stmt     ::= small_stmt simple_tail
simple_tail     ::= SEMICOLON small_stmt simple_tail
                  | SEMICOLON
                  | empty
small_stmt      ::= import_stmt
                  | assignment_stmt
                  | annotated_assignment
                  | return_stmt
                  | flow_stmt
                  | assert_stmt
                  | del_stmt
                  | global_stmt
                  | nonlocal_stmt
                  | expression_stmt
                  | recovery_stmt
```

## Imports And Definitions

```bnf
import_stmt     ::= IMPORT import_item import_item_tail
                  | FROM dotted_name IMPORT import_target import_target_tail
import_item_tail
                ::= COMMA import_item import_item_tail
                  | empty
import_target_tail
                ::= COMMA import_target import_target_tail
                  | empty
import_item     ::= dotted_name import_alias_opt
import_target   ::= IDENTIFIER import_alias_opt
                  | STAR
import_alias_opt
                ::= AS IDENTIFIER
                  | empty

decorated_def   ::= decorator_list function_def
decorator_list  ::= decorator decorator_list
                  | decorator
decorator       ::= AT dotted_name call_args_opt NEWLINE

compound_stmt   ::= decorated_def
                  | function_def
                  | async_function_def
                  | class_def
                  | if_stmt
                  | async_for_stmt
                  | for_stmt
                  | while_stmt
                  | try_stmt
                  | async_with_stmt
                  | with_stmt

function_def    ::= DEF IDENTIFIER LPAREN parameter_list_opt RPAREN return_type_opt COLON suite
async_function_def
                ::= ASYNC function_def
class_def       ::= CLASS IDENTIFIER class_bases_opt COLON suite
class_bases_opt ::= call_args
                  | empty
return_type_opt ::= ARROW expression
                  | empty
parameter_list_opt
                ::= parameter_list
                  | empty
parameter_list  ::= parameter parameter_tail trailing_comma_opt
parameter_tail  ::= COMMA parameter parameter_tail
                  | empty
parameter       ::= STAR
                  | STAR IDENTIFIER
                  | POWER IDENTIFIER
                  | IDENTIFIER annotation_opt default_opt
annotation_opt  ::= COLON expression
                  | empty
default_opt     ::= ASSIGN expression
                  | empty
trailing_comma_opt
                ::= COMMA
                  | empty
```

## Blocks And Control Flow

```bnf
suite           ::= simple_stmt NEWLINE
                  | NEWLINE INDENT statement_list DEDENT

if_stmt         ::= IF expression COLON suite elif_list else_opt
elif_list       ::= ELIF expression COLON suite elif_list
                  | empty
else_opt        ::= ELSE COLON suite
                  | empty

async_for_stmt  ::= ASYNC for_stmt
for_stmt        ::= FOR target_list IN expression COLON suite else_opt
while_stmt      ::= WHILE expression COLON suite else_opt

try_stmt        ::= TRY COLON suite except_list finally_opt
except_list     ::= EXCEPT expression_opt COLON suite except_list
                  | empty
finally_opt     ::= FINALLY COLON suite
                  | empty

async_with_stmt ::= ASYNC with_stmt
with_stmt       ::= WITH with_item with_item_tail COLON suite
with_item_tail  ::= COMMA with_item with_item_tail
                  | empty
with_item       ::= expression as_target_opt
as_target_opt   ::= AS target
                  | empty
```

## Statements

```bnf
assignment_stmt ::= target_list assign_op expression
annotated_assignment
                ::= target COLON expression default_opt
assign_op       ::= ASSIGN
                  | PLUS_ASSIGN | MINUS_ASSIGN | STAR_ASSIGN | SLASH_ASSIGN
                  | PERCENT_ASSIGN | AMP_ASSIGN | PIPE_ASSIGN | CARET_ASSIGN
                  | AT_ASSIGN | LSHIFT_ASSIGN | RSHIFT_ASSIGN
                  | POWER_ASSIGN | FLOORDIV_ASSIGN
return_stmt     ::= RETURN expression_opt
flow_stmt       ::= PASS
                  | BREAK
                  | CONTINUE
                  | RAISE expression_opt
                  | yield_stmt
yield_stmt      ::= YIELD yield_value_opt
yield_value_opt ::= expression
                  | FROM expression
                  | empty
assert_stmt     ::= ASSERT expression assert_message_opt
assert_message_opt
                ::= COMMA expression
                  | empty
del_stmt        ::= DEL target_list
global_stmt     ::= GLOBAL name_list
nonlocal_stmt   ::= NONLOCAL name_list
name_list       ::= IDENTIFIER name_tail
name_tail       ::= COMMA IDENTIFIER name_tail
                  | empty
expression_stmt ::= expression
recovery_stmt   ::= ERROR
                  | BACKSLASH

target_list     ::= target target_tail
target_tail     ::= COMMA target target_tail
                  | COMMA
                  | empty
target          ::= IDENTIFIER
                  | attribute
                  | subscript
                  | LPAREN target_list RPAREN
                  | LBRACKET target_list RBRACKET
```

## Expressions

```bnf
expression_opt  ::= expression
                  | empty
expression      ::= named_expr
named_expr      ::= lambda_expr
                  | conditional_expr named_tail
named_tail      ::= WALRUS expression
                  | empty
lambda_expr     ::= LAMBDA parameter_list_opt COLON expression
conditional_expr
                ::= or_expr if_else_opt
if_else_opt     ::= IF expression ELSE expression
                  | empty

or_expr         ::= and_expr or_tail
or_tail         ::= OR and_expr or_tail
                  | empty
and_expr        ::= not_expr and_tail
and_tail        ::= AND not_expr and_tail
                  | empty
not_expr        ::= NOT not_expr
                  | comparison

comparison      ::= bit_or_expr comparison_tail
comparison_tail ::= compare_op bit_or_expr comparison_tail
                  | empty
compare_op      ::= EQ | NE | LT | LE | GT | GE
                  | IS
                  | IS NOT
                  | IN
                  | NOT IN

bit_or_expr     ::= bit_xor_expr bit_or_tail
bit_or_tail     ::= PIPE bit_xor_expr bit_or_tail
                  | empty
bit_xor_expr    ::= bit_and_expr bit_xor_tail
bit_xor_tail    ::= CARET bit_and_expr bit_xor_tail
                  | empty
bit_and_expr    ::= shift_expr bit_and_tail
bit_and_tail    ::= AMP shift_expr bit_and_tail
                  | empty
shift_expr      ::= arith_expr shift_tail
shift_tail      ::= LSHIFT arith_expr shift_tail
                  | RSHIFT arith_expr shift_tail
                  | empty
arith_expr      ::= term arith_tail
arith_tail      ::= PLUS term arith_tail
                  | MINUS term arith_tail
                  | empty
term            ::= factor term_tail
term_tail       ::= STAR factor term_tail
                  | SLASH factor term_tail
                  | FLOORDIV factor term_tail
                  | PERCENT factor term_tail
                  | AT factor term_tail
                  | empty
factor          ::= AWAIT factor
                  | PLUS factor
                  | MINUS factor
                  | TILDE factor
                  | power
power           ::= primary power_tail
power_tail      ::= POWER factor
                  | empty
```

## Primaries, Calls, Attributes, And Launches

```bnf
primary         ::= atom trailer_list
trailer_list    ::= trailer trailer_list
                  | empty
trailer         ::= call_args
                  | LBRACKET subscript_expr RBRACKET
                  | DOT IDENTIFIER

call_args_opt   ::= call_args
                  | empty
call_args       ::= LPAREN argument_list_opt RPAREN
argument_list_opt
                ::= argument_list
                  | empty
argument_list   ::= argument argument_tail trailing_comma_opt
argument_tail   ::= COMMA argument argument_tail
                  | empty
argument        ::= expression
                  | IDENTIFIER ASSIGN expression
                  | STAR expression
                  | POWER expression

subscript_expr  ::= expression slice_tail_opt
                  | slice_expr
slice_tail_opt  ::= COLON expression_opt slice_step_opt
                  | empty
slice_expr      ::= expression_opt COLON expression_opt slice_step_opt
slice_step_opt  ::= COLON expression_opt
                  | empty

atom            ::= IDENTIFIER
                  | INTEGER
                  | FLOAT
                  | COMPLEX
                  | STRING
                  | ELLIPSIS
                  | NONE
                  | TRUE
                  | FALSE
                  | tuple_expr
                  | list_expr
                  | dict_expr
tuple_expr      ::= LPAREN expression_list_opt RPAREN
list_expr       ::= LBRACKET expression_list_opt RBRACKET
dict_expr       ::= LBRACE dict_items_opt RBRACE
expression_list_opt
                ::= expression_list
                  | empty
expression_list ::= expression expression_tail trailing_comma_opt
expression_tail ::= COMMA expression expression_tail
                  | empty
dict_items_opt  ::= dict_items
                  | empty
dict_items      ::= dict_item dict_tail trailing_comma_opt
dict_tail       ::= COMMA dict_item dict_tail
                  | empty
dict_item       ::= expression COLON expression
```

`_kernel[grid](args...)` does not need a special lexer token. It is parsed as:

```bnf
kernel_launch_shape ::= primary LBRACKET expression RBRACKET call_args
```

The semantic pass can later decide whether that primary names a Triton kernel.

## Current Yacc Parser Iteration

The implemented yacc parser in `parser/triton_parser.y` is intentionally simpler
than the full CFG above. Its current job is to validate:

- Python-style logical lines and indentation blocks.
- End-of-file with or without a final `NEWLINE`.
- Indented blocks introduced by a block header that ends in a top-level colon.
- Balanced `LPAREN`/`RPAREN`, `LBRACKET`/`RBRACKET`, and
  `LBRACE`/`RBRACE` inside each logical line.
- Full token coverage: every lexer token appears in the parser source.

This keeps the first executable parser useful without pretending to enforce all
Python/Triton syntax yet. Future iterations should replace more generic
`line_items` usage with productions from the richer CFG above.

Run `python3 parser/run_smoke_tests.py` after building the parser to confirm it
accepts simple valid cases, rejects malformed delimiter structure, and rejects
indentation after a non-header line.

## Dotted Names

```bnf
dotted_name     ::= IDENTIFIER dotted_tail
dotted_tail     ::= DOT IDENTIFIER dotted_tail
                  | empty
attribute       ::= primary DOT IDENTIFIER
subscript       ::= primary LBRACKET subscript_expr RBRACKET
```

Examples represented by this rule:

```text
triton.jit
triton.language
tl.constexpr
tl.load
torch.empty_like
torch.nn.functional
```

## Initial Incremental Checks

The first analyzer does not need to prove full correctness. A useful first
semantic layer can check:

- Imports: record aliases for `triton`, `triton.language`, `torch`, and
  `torch.nn.functional`.
- Decorators: identify functions decorated with `triton.jit` through the alias
  table, not by hard-coded token names.
- Function signatures: collect parameter names, defaults, annotations, `*`, and
  `**` parameters.
- Variable definitions: record assignment targets and warn on obvious reads of
  names that have not been defined in the current function or parameter list.
- Kernel body restrictions: inside a `triton.jit` function, warn on direct
  `torch.*` calls or non-Triton helper calls.
- Triton meta parameters: detect parameters annotated as `tl.constexpr`.
- Launch syntax: detect `_kernel[grid](...)` forms and verify that the launched
  name resolves to a known `triton.jit` function when available.
- Core block sanity: rely on `INDENT` and `DEDENT` tokens to verify block
  structure before deeper semantic checks.
