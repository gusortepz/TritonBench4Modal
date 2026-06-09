/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     T_ERROR = 258,
     T_NEWLINE = 260,
     T_INDENT = 261,
     T_DEDENT = 262,
     T_IDENTIFIER = 263,
     T_INTEGER = 264,
     T_FLOAT = 265,
     T_COMPLEX = 266,
     T_STRING = 267,
     T_DEF = 268,
     T_RETURN = 269,
     T_IF = 270,
     T_ELIF = 271,
     T_ELSE = 272,
     T_FOR = 273,
     T_WHILE = 274,
     T_IN = 275,
     T_IMPORT = 276,
     T_FROM = 277,
     T_AS = 278,
     T_TRY = 279,
     T_EXCEPT = 280,
     T_FINALLY = 281,
     T_PASS = 282,
     T_BREAK = 283,
     T_CONTINUE = 284,
     T_CLASS = 285,
     T_WITH = 286,
     T_LAMBDA = 287,
     T_YIELD = 288,
     T_ASSERT = 289,
     T_RAISE = 290,
     T_GLOBAL = 291,
     T_NONLOCAL = 292,
     T_DEL = 293,
     T_ASYNC = 294,
     T_AWAIT = 295,
     T_AND = 296,
     T_OR = 297,
     T_NOT = 298,
     T_IS = 299,
     T_NONE = 300,
     T_TRUE = 301,
     T_FALSE = 302,
     T_ARROW = 303,
     T_WALRUS = 304,
     T_ELLIPSIS = 305,
     T_EQ = 306,
     T_NE = 307,
     T_LE = 308,
     T_GE = 309,
     T_LSHIFT = 310,
     T_RSHIFT = 311,
     T_POWER = 312,
     T_FLOORDIV = 313,
     T_PLUS_ASSIGN = 314,
     T_MINUS_ASSIGN = 315,
     T_STAR_ASSIGN = 316,
     T_SLASH_ASSIGN = 317,
     T_PERCENT_ASSIGN = 318,
     T_AMP_ASSIGN = 319,
     T_PIPE_ASSIGN = 320,
     T_CARET_ASSIGN = 321,
     T_AT_ASSIGN = 322,
     T_LSHIFT_ASSIGN = 323,
     T_RSHIFT_ASSIGN = 324,
     T_POWER_ASSIGN = 325,
     T_FLOORDIV_ASSIGN = 326,
     T_ASSIGN = 327,
     T_LT = 328,
     T_GT = 329,
     T_PLUS = 330,
     T_MINUS = 331,
     T_STAR = 332,
     T_SLASH = 333,
     T_PERCENT = 334,
     T_AMP = 335,
     T_PIPE = 336,
     T_CARET = 337,
     T_TILDE = 338,
     T_DOT = 339,
     T_COMMA = 340,
     T_COLON = 341,
     T_SEMICOLON = 342,
     T_AT = 343,
     T_BACKSLASH = 344,
     T_LPAREN = 345,
     T_RPAREN = 346,
     T_LBRACKET = 347,
     T_RBRACKET = 348,
     T_LBRACE = 349,
     T_RBRACE = 350
   };
#endif
/* Tokens.  */
#define T_ERROR 258
#define T_NEWLINE 260
#define T_INDENT 261
#define T_DEDENT 262
#define T_IDENTIFIER 263
#define T_INTEGER 264
#define T_FLOAT 265
#define T_COMPLEX 266
#define T_STRING 267
#define T_DEF 268
#define T_RETURN 269
#define T_IF 270
#define T_ELIF 271
#define T_ELSE 272
#define T_FOR 273
#define T_WHILE 274
#define T_IN 275
#define T_IMPORT 276
#define T_FROM 277
#define T_AS 278
#define T_TRY 279
#define T_EXCEPT 280
#define T_FINALLY 281
#define T_PASS 282
#define T_BREAK 283
#define T_CONTINUE 284
#define T_CLASS 285
#define T_WITH 286
#define T_LAMBDA 287
#define T_YIELD 288
#define T_ASSERT 289
#define T_RAISE 290
#define T_GLOBAL 291
#define T_NONLOCAL 292
#define T_DEL 293
#define T_ASYNC 294
#define T_AWAIT 295
#define T_AND 296
#define T_OR 297
#define T_NOT 298
#define T_IS 299
#define T_NONE 300
#define T_TRUE 301
#define T_FALSE 302
#define T_ARROW 303
#define T_WALRUS 304
#define T_ELLIPSIS 305
#define T_EQ 306
#define T_NE 307
#define T_LE 308
#define T_GE 309
#define T_LSHIFT 310
#define T_RSHIFT 311
#define T_POWER 312
#define T_FLOORDIV 313
#define T_PLUS_ASSIGN 314
#define T_MINUS_ASSIGN 315
#define T_STAR_ASSIGN 316
#define T_SLASH_ASSIGN 317
#define T_PERCENT_ASSIGN 318
#define T_AMP_ASSIGN 319
#define T_PIPE_ASSIGN 320
#define T_CARET_ASSIGN 321
#define T_AT_ASSIGN 322
#define T_LSHIFT_ASSIGN 323
#define T_RSHIFT_ASSIGN 324
#define T_POWER_ASSIGN 325
#define T_FLOORDIV_ASSIGN 326
#define T_ASSIGN 327
#define T_LT 328
#define T_GT 329
#define T_PLUS 330
#define T_MINUS 331
#define T_STAR 332
#define T_SLASH 333
#define T_PERCENT 334
#define T_AMP 335
#define T_PIPE 336
#define T_CARET 337
#define T_TILDE 338
#define T_DOT 339
#define T_COMMA 340
#define T_COLON 341
#define T_SEMICOLON 342
#define T_AT 343
#define T_BACKSLASH 344
#define T_LPAREN 345
#define T_RPAREN 346
#define T_LBRACKET 347
#define T_RBRACKET 348
#define T_LBRACE 349
#define T_RBRACE 350




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

