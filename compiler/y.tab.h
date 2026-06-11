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
     NAME = 258,
     INTEGER = 259,
     FLOAT = 260,
     STRING = 261,
     DEF = 262,
     RETURN = 263,
     IF = 264,
     ELIF = 265,
     ELSE = 266,
     FOR = 267,
     WHILE = 268,
     IN = 269,
     AND = 270,
     OR = 271,
     NOT = 272,
     IS = 273,
     TRUE = 274,
     FALSE = 275,
     NONE = 276,
     IMPORT = 277,
     FROM = 278,
     AS = 279,
     PASS = 280,
     RAISE = 281,
     ASSERT = 282,
     BREAK = 283,
     CONTINUE = 284,
     GLOBAL = 285,
     NONLOCAL = 286,
     DEL = 287,
     PLUS = 288,
     MINUS = 289,
     STAR = 290,
     DOUBLE_STAR = 291,
     SLASH = 292,
     DOUBLE_SLASH = 293,
     PERCENT = 294,
     AT = 295,
     LESS = 296,
     LESS_EQUAL = 297,
     GREATER = 298,
     GREATER_EQUAL = 299,
     EQUAL_EQUAL = 300,
     NOT_EQUAL = 301,
     AMPERSAND = 302,
     PIPE = 303,
     CARET = 304,
     TILDE = 305,
     LEFT_SHIFT = 306,
     RIGHT_SHIFT = 307,
     ASSIGN = 308,
     PLUS_ASSIGN = 309,
     MINUS_ASSIGN = 310,
     STAR_ASSIGN = 311,
     SLASH_ASSIGN = 312,
     PERCENT_ASSIGN = 313,
     AMPERSAND_ASSIGN = 314,
     PIPE_ASSIGN = 315,
     CARET_ASSIGN = 316,
     LEFT_SHIFT_ASSIGN = 317,
     RIGHT_SHIFT_ASSIGN = 318,
     DOUBLE_STAR_ASSIGN = 319,
     DOUBLE_SLASH_ASSIGN = 320,
     OPEN_PARENTHESIS = 321,
     CLOSE_PARENTHESIS = 322,
     OPEN_BRACKET = 323,
     CLOSE_BRACKET = 324,
     OPEN_BRACE = 325,
     CLOSE_BRACE = 326,
     COMMA = 327,
     COLON = 328,
     DOT = 329,
     SEMICOLON = 330,
     ARROW = 331,
     ELLIPSIS = 332,
     ERROR = 333
   };
#endif
/* Tokens.  */
#define NAME 258
#define INTEGER 259
#define FLOAT 260
#define STRING 261
#define DEF 262
#define RETURN 263
#define IF 264
#define ELIF 265
#define ELSE 266
#define FOR 267
#define WHILE 268
#define IN 269
#define AND 270
#define OR 271
#define NOT 272
#define IS 273
#define TRUE 274
#define FALSE 275
#define NONE 276
#define IMPORT 277
#define FROM 278
#define AS 279
#define PASS 280
#define RAISE 281
#define ASSERT 282
#define BREAK 283
#define CONTINUE 284
#define GLOBAL 285
#define NONLOCAL 286
#define DEL 287
#define PLUS 288
#define MINUS 289
#define STAR 290
#define DOUBLE_STAR 291
#define SLASH 292
#define DOUBLE_SLASH 293
#define PERCENT 294
#define AT 295
#define LESS 296
#define LESS_EQUAL 297
#define GREATER 298
#define GREATER_EQUAL 299
#define EQUAL_EQUAL 300
#define NOT_EQUAL 301
#define AMPERSAND 302
#define PIPE 303
#define CARET 304
#define TILDE 305
#define LEFT_SHIFT 306
#define RIGHT_SHIFT 307
#define ASSIGN 308
#define PLUS_ASSIGN 309
#define MINUS_ASSIGN 310
#define STAR_ASSIGN 311
#define SLASH_ASSIGN 312
#define PERCENT_ASSIGN 313
#define AMPERSAND_ASSIGN 314
#define PIPE_ASSIGN 315
#define CARET_ASSIGN 316
#define LEFT_SHIFT_ASSIGN 317
#define RIGHT_SHIFT_ASSIGN 318
#define DOUBLE_STAR_ASSIGN 319
#define DOUBLE_SLASH_ASSIGN 320
#define OPEN_PARENTHESIS 321
#define CLOSE_PARENTHESIS 322
#define OPEN_BRACKET 323
#define CLOSE_BRACKET 324
#define OPEN_BRACE 325
#define CLOSE_BRACE 326
#define COMMA 327
#define COLON 328
#define DOT 329
#define SEMICOLON 330
#define ARROW 331
#define ELLIPSIS 332
#define ERROR 333




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

