/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



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




/* Copy the first part of user declarations.  */
#line 1 "newparser.y"

#include <stdio.h>
#include <stdlib.h>

int yylex(void);
void yyerror(const char *msg);

extern int yylineno;
extern char *yytext;
/* set by the lexer; triton requires BOTH import triton and @triton.jit */
extern int triton_import_seen;
extern int triton_jit_seen;


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 276 "y.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  94
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   312

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  79
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  94
/* YYNRULES -- Number of rules.  */
#define YYNRULES  208
/* YYNRULES -- Number of states.  */
#define YYNSTATES  311

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   333

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     6,     7,     9,    11,    15,    19,    22,
      26,    32,    34,    36,    42,    45,    46,    50,    54,    55,
      59,    61,    62,    65,    68,    69,    72,    73,    76,    80,
      83,    84,    87,    88,    91,    94,    96,    98,   100,   102,
     106,   109,   112,   115,   118,   122,   125,   126,   130,   131,
     133,   135,   137,   139,   141,   143,   145,   147,   149,   151,
     153,   155,   157,   158,   161,   162,   164,   165,   168,   172,
     173,   176,   181,   183,   185,   188,   192,   193,   196,   199,
     203,   204,   207,   210,   211,   214,   218,   219,   221,   224,
     227,   228,   231,   232,   235,   238,   239,   242,   243,   246,
     250,   251,   254,   258,   259,   262,   264,   267,   271,   272,
     274,   276,   278,   280,   282,   284,   286,   289,   292,   294,
     295,   298,   302,   303,   306,   310,   311,   314,   318,   319,
     322,   326,   330,   331,   334,   338,   342,   343,   346,   350,
     354,   358,   362,   366,   367,   370,   373,   376,   378,   381,
     384,   385,   388,   391,   392,   396,   400,   403,   405,   406,
     408,   410,   412,   414,   416,   418,   420,   422,   426,   430,
     434,   436,   437,   439,   440,   442,   443,   446,   449,   450,
     453,   456,   457,   460,   461,   465,   468,   471,   472,   475,
     476,   479,   482,   483,   486,   489,   490,   493,   494,   497,
     500,   503,   504,   507,   510,   511,   514,   515,   517
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      80,     0,    -1,    81,    80,    -1,    -1,    82,    -1,    95,
      -1,     9,   115,    73,    -1,    10,   115,    73,    -1,    11,
      73,    -1,    13,   115,    73,    -1,    12,   119,    14,   116,
      73,    -1,    83,    -1,    85,    -1,     7,     3,    87,    84,
      73,    -1,    76,   115,    -1,    -1,    40,   113,    86,    -1,
      66,   149,    67,    -1,    -1,    66,    88,    67,    -1,    89,
      -1,    -1,    92,    90,    -1,    72,    91,    -1,    -1,    92,
      90,    -1,    -1,     3,    93,    -1,    73,   115,    94,    -1,
      53,   115,    -1,    -1,    53,   115,    -1,    -1,   116,    96,
      -1,     8,    99,    -1,    25,    -1,    28,    -1,    29,    -1,
     104,    -1,    27,   115,   100,    -1,    26,   101,    -1,    30,
     102,    -1,    31,   102,    -1,    32,   119,    -1,    53,   116,
      97,    -1,    98,   116,    -1,    -1,    53,   116,    97,    -1,
      -1,    54,    -1,    55,    -1,    56,    -1,    57,    -1,    65,
      -1,    58,    -1,    64,    -1,    59,    -1,    60,    -1,    61,
      -1,    62,    -1,    63,    -1,   116,    -1,    -1,    72,   115,
      -1,    -1,   115,    -1,    -1,     3,   103,    -1,    72,     3,
     103,    -1,    -1,    22,   106,    -1,    23,   113,    22,   105,
      -1,    35,    -1,   109,    -1,   108,   107,    -1,    72,   108,
     107,    -1,    -1,   113,   112,    -1,   111,   110,    -1,    72,
     111,   110,    -1,    -1,     3,   112,    -1,    24,     3,    -1,
      -1,     3,   114,    -1,    74,     3,   114,    -1,    -1,   122,
      -1,   115,   117,    -1,    72,   118,    -1,    -1,   115,   117,
      -1,    -1,   131,   120,    -1,    72,   121,    -1,    -1,   131,
     120,    -1,    -1,   124,   123,    -1,    16,   124,   123,    -1,
      -1,   126,   125,    -1,    15,   126,   125,    -1,    -1,    17,
     126,    -1,   127,    -1,   131,   128,    -1,   129,   131,   128,
      -1,    -1,    41,    -1,    42,    -1,    43,    -1,    44,    -1,
      45,    -1,    46,    -1,    14,    -1,    17,    14,    -1,    18,
     130,    -1,    17,    -1,    -1,   133,   132,    -1,    48,   133,
     132,    -1,    -1,   135,   134,    -1,    49,   135,   134,    -1,
      -1,   137,   136,    -1,    47,   137,   136,    -1,    -1,   139,
     138,    -1,    51,   139,   138,    -1,    52,   139,   138,    -1,
      -1,   141,   140,    -1,    33,   141,   140,    -1,    34,   141,
     140,    -1,    -1,   143,   142,    -1,    35,   143,   142,    -1,
      37,   143,   142,    -1,    38,   143,   142,    -1,    39,   143,
     142,    -1,    40,   143,   142,    -1,    -1,    33,   143,    -1,
      34,   143,    -1,    50,   143,    -1,   144,    -1,   146,   145,
      -1,    36,   143,    -1,    -1,   150,   147,    -1,   148,   147,
      -1,    -1,    66,   149,    67,    -1,    68,   165,    69,    -1,
      74,     3,    -1,   160,    -1,    -1,     3,    -1,     4,    -1,
       5,    -1,   154,    -1,    19,    -1,    20,    -1,    21,    -1,
      77,    -1,    66,   151,    67,    -1,    68,   152,    69,    -1,
      70,   153,    71,    -1,   116,    -1,    -1,   116,    -1,    -1,
     156,    -1,    -1,     6,   155,    -1,     6,   155,    -1,    -1,
     159,   157,    -1,    72,   158,    -1,    -1,   159,   157,    -1,
      -1,   115,    73,   115,    -1,   163,   161,    -1,    72,   162,
      -1,    -1,   163,   161,    -1,    -1,   115,   164,    -1,    53,
     115,    -1,    -1,   168,   166,    -1,    72,   167,    -1,    -1,
     168,   166,    -1,    -1,   115,   169,    -1,    73,   170,    -1,
      73,   170,    -1,    -1,   115,   171,    -1,    73,   172,    -1,
      -1,    73,   172,    -1,    -1,   115,    -1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    50,    50,    51,    55,    56,    64,    65,    66,    67,
      68,    69,    70,    74,    78,    79,    83,    87,    88,    96,
     100,   101,   105,   109,   110,   114,   115,   119,   123,   124,
     125,   129,   130,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   154,   155,   156,   160,   161,   165,
     165,   165,   165,   166,   166,   166,   167,   167,   167,   168,
     168,   174,   175,   179,   180,   184,   185,   189,   193,   194,
     200,   201,   205,   206,   210,   214,   215,   219,   223,   227,
     228,   232,   236,   237,   241,   245,   246,   255,   261,   265,
     266,   270,   271,   276,   280,   281,   285,   286,   292,   296,
     297,   301,   305,   306,   310,   311,   315,   319,   320,   324,
     324,   324,   324,   325,   325,   326,   327,   328,   332,   333,
     337,   341,   342,   346,   350,   351,   355,   359,   360,   364,
     368,   369,   370,   374,   378,   379,   380,   384,   388,   389,
     390,   391,   392,   393,   397,   398,   399,   400,   404,   408,
     409,   415,   419,   420,   424,   425,   426,   430,   431,   435,
     436,   437,   438,   439,   440,   441,   442,   443,   444,   445,
     449,   450,   454,   455,   459,   460,   464,   468,   469,   473,
     477,   478,   482,   483,   487,   493,   497,   498,   502,   503,
     507,   511,   512,   518,   522,   523,   527,   528,   532,   533,
     537,   538,   542,   543,   544,   548,   549,   553,   554
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "NAME", "INTEGER", "FLOAT", "STRING",
  "DEF", "RETURN", "IF", "ELIF", "ELSE", "FOR", "WHILE", "IN", "AND", "OR",
  "NOT", "IS", "TRUE", "FALSE", "NONE", "IMPORT", "FROM", "AS", "PASS",
  "RAISE", "ASSERT", "BREAK", "CONTINUE", "GLOBAL", "NONLOCAL", "DEL",
  "PLUS", "MINUS", "STAR", "DOUBLE_STAR", "SLASH", "DOUBLE_SLASH",
  "PERCENT", "AT", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL",
  "EQUAL_EQUAL", "NOT_EQUAL", "AMPERSAND", "PIPE", "CARET", "TILDE",
  "LEFT_SHIFT", "RIGHT_SHIFT", "ASSIGN", "PLUS_ASSIGN", "MINUS_ASSIGN",
  "STAR_ASSIGN", "SLASH_ASSIGN", "PERCENT_ASSIGN", "AMPERSAND_ASSIGN",
  "PIPE_ASSIGN", "CARET_ASSIGN", "LEFT_SHIFT_ASSIGN", "RIGHT_SHIFT_ASSIGN",
  "DOUBLE_STAR_ASSIGN", "DOUBLE_SLASH_ASSIGN", "OPEN_PARENTHESIS",
  "CLOSE_PARENTHESIS", "OPEN_BRACKET", "CLOSE_BRACKET", "OPEN_BRACE",
  "CLOSE_BRACE", "COMMA", "COLON", "DOT", "SEMICOLON", "ARROW", "ELLIPSIS",
  "ERROR", "$accept", "program", "element", "header", "funcheader",
  "ret_annot", "decorator", "deco_args", "parameters", "params",
  "paramlist", "paramlist_tail", "param_after_comma", "param",
  "param_tail", "param_default", "stmt", "assign_tail", "eq_chain_tail",
  "augassign", "return_tail", "assert_tail", "raise_tail", "name_list",
  "name_list_tail", "import_stmt", "import_target", "dotted_as_names",
  "dotted_as_names_tail", "dotted_as_name", "name_as_names",
  "name_as_names_tail", "name_as_name", "as_opt", "dotted_name",
  "dotted_name_tail", "test", "testlist", "testlist_tail",
  "testlist_after_comma", "exprlist", "exprlist_tail",
  "exprlist_after_comma", "or_test", "or_tail", "and_test", "and_tail",
  "not_test", "comparison", "comp_tail", "comp_op", "is_not_opt", "expr",
  "bitor_tail", "xor_expr", "xor_tail", "and_expr", "bitand_tail",
  "shift_expr", "shift_tail", "arith_expr", "arith_tail", "term",
  "term_tail", "factor", "power", "power_tail", "atom_expr",
  "trailer_list", "trailer", "call_args", "atom", "paren_body",
  "list_body", "dict_body", "strings", "strings_tail", "dict_items",
  "dict_items_tail", "dict_items_after_comma", "dict_item", "arglist",
  "arglist_tail", "arglist_after_comma", "argument", "kwarg_tail",
  "subscriptlist", "subs_tail", "subs_after_comma", "subscript",
  "subscript_tail", "slice_upper", "slice_step_opt", "step_opt", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    79,    80,    80,    81,    81,    82,    82,    82,    82,
      82,    82,    82,    83,    84,    84,    85,    86,    86,    87,
      88,    88,    89,    90,    90,    91,    91,    92,    93,    93,
      93,    94,    94,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    96,    96,    96,    97,    97,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    99,    99,   100,   100,   101,   101,   102,   103,   103,
     104,   104,   105,   105,   106,   107,   107,   108,   109,   110,
     110,   111,   112,   112,   113,   114,   114,   115,   116,   117,
     117,   118,   118,   119,   120,   120,   121,   121,   122,   123,
     123,   124,   125,   125,   126,   126,   127,   128,   128,   129,
     129,   129,   129,   129,   129,   129,   129,   129,   130,   130,
     131,   132,   132,   133,   134,   134,   135,   136,   136,   137,
     138,   138,   138,   139,   140,   140,   140,   141,   142,   142,
     142,   142,   142,   142,   143,   143,   143,   143,   144,   145,
     145,   146,   147,   147,   148,   148,   148,   149,   149,   150,
     150,   150,   150,   150,   150,   150,   150,   150,   150,   150,
     151,   151,   152,   152,   153,   153,   154,   155,   155,   156,
     157,   157,   158,   158,   159,   160,   161,   161,   162,   162,
     163,   164,   164,   165,   166,   166,   167,   167,   168,   168,
     169,   169,   170,   170,   170,   171,   171,   172,   172
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     2,     0,     1,     1,     3,     3,     2,     3,
       5,     1,     1,     5,     2,     0,     3,     3,     0,     3,
       1,     0,     2,     2,     0,     2,     0,     2,     3,     2,
       0,     2,     0,     2,     2,     1,     1,     1,     1,     3,
       2,     2,     2,     2,     3,     2,     0,     3,     0,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     0,     2,     0,     1,     0,     2,     3,     0,
       2,     4,     1,     1,     2,     3,     0,     2,     2,     3,
       0,     2,     2,     0,     2,     3,     0,     1,     2,     2,
       0,     2,     0,     2,     2,     0,     2,     0,     2,     3,
       0,     2,     3,     0,     2,     1,     2,     3,     0,     1,
       1,     1,     1,     1,     1,     1,     2,     2,     1,     0,
       2,     3,     0,     2,     3,     0,     2,     3,     0,     2,
       3,     3,     0,     2,     3,     3,     0,     2,     3,     3,
       3,     3,     3,     0,     2,     2,     2,     1,     2,     2,
       0,     2,     2,     0,     3,     3,     2,     1,     0,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       1,     0,     1,     0,     1,     0,     2,     2,     0,     2,
       2,     0,     2,     0,     3,     2,     2,     0,     2,     0,
       2,     2,     0,     2,     2,     0,     2,     0,     2,     2,
       2,     0,     2,     2,     0,     2,     0,     1,     0
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,   159,   160,   161,   178,     0,    62,     0,     0,     0,
       0,     0,     0,   163,   164,   165,     0,     0,    35,    66,
       0,    36,    37,     0,     0,     0,     0,     0,     0,     0,
     171,   173,   175,   166,     0,     3,     4,    11,    12,     5,
      38,    90,    46,    87,   100,   103,   105,   108,   122,   125,
     128,   132,   136,   143,   147,   150,   153,   162,   178,   176,
       0,    34,    61,     0,     0,     8,     0,    95,     0,   104,
      86,    70,    76,    83,     0,    40,    65,    64,    69,    41,
      42,    43,   144,   145,    18,   146,   170,     0,   172,     0,
       0,     0,   174,   181,     1,     2,    92,    88,     0,    49,
      50,    51,    52,    54,    56,    57,    58,    59,    60,    55,
      53,    33,     0,     0,    98,     0,   101,   115,     0,   119,
     109,   110,   111,   112,   113,   114,   106,     0,     0,   120,
       0,   123,     0,   126,     0,     0,   129,     0,     0,   133,
       0,     0,     0,     0,     0,   137,     0,   148,   158,     0,
       0,   151,   153,   177,    21,    15,     6,     7,     0,    97,
      93,     9,     0,    84,     0,    74,     0,    77,     0,     0,
      39,     0,    67,   158,    16,   167,   168,     0,   169,   183,
     179,    90,    89,    48,    45,   100,   103,   116,   118,   117,
     108,   122,   125,   128,   132,   132,   136,   136,   143,   143,
     143,   143,   143,   149,   192,     0,   157,   187,   204,   201,
       0,   195,   156,   152,    30,     0,    20,    24,     0,     0,
       0,    94,    95,    86,    76,    82,    83,    72,    71,    73,
      80,    63,    69,     0,   184,   180,   181,    91,     0,    44,
      99,   102,   107,   121,   124,   127,   130,   131,   134,   135,
     138,   139,   140,   141,   142,     0,   190,   154,   189,   185,
     208,   206,   199,   204,   198,   155,   197,   193,     0,     0,
      27,    19,    26,    22,    14,    13,    10,    96,    85,    75,
      81,     0,    78,    68,    17,   182,    48,   191,   186,   187,
     207,   203,   208,   202,   200,   194,   195,    29,    32,    23,
      24,    80,    47,   188,   205,   196,     0,    28,    25,    79,
      31
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    34,    35,    36,    37,   219,    38,   174,   155,   215,
     216,   273,   299,   217,   270,   307,    39,   111,   239,   112,
      61,   170,    75,    79,   172,    40,   228,    71,   165,    72,
     229,   282,   230,   167,    73,   163,    41,    42,    97,   182,
      66,   160,   221,    43,   114,    44,   116,    45,    46,   126,
     127,   189,    47,   129,    48,   131,    49,   133,    50,   136,
      51,   139,    52,   145,    53,    54,   147,    55,   151,   152,
     205,    56,    87,    89,    91,    57,    59,    92,   180,   235,
      93,   206,   259,   288,   207,   256,   210,   267,   295,   211,
     264,   262,   293,   291
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -134
static const yytype_int16 yypact[] =
{
      25,  -134,  -134,  -134,     8,    13,   199,   199,   199,   -55,
     221,   199,   199,  -134,  -134,  -134,    21,    21,  -134,   199,
     199,  -134,  -134,    37,    37,   221,   221,   221,    21,   221,
     199,   199,   199,  -134,    43,    25,  -134,  -134,  -134,  -134,
    -134,     9,   247,  -134,    69,    71,  -134,   134,    40,    38,
      45,   -41,   -13,    39,  -134,    54,   -51,  -134,     8,  -134,
      33,  -134,  -134,    27,    28,  -134,    84,    31,    34,  -134,
      30,  -134,    55,    81,    87,  -134,  -134,    56,    58,  -134,
    -134,  -134,  -134,  -134,    44,  -134,  -134,    59,  -134,    42,
      61,    41,  -134,    60,  -134,  -134,   199,  -134,   199,  -134,
    -134,  -134,  -134,  -134,  -134,  -134,  -134,  -134,  -134,  -134,
    -134,  -134,   199,   199,  -134,   199,  -134,  -134,   121,   120,
    -134,  -134,  -134,  -134,  -134,  -134,  -134,   221,   221,  -134,
     221,  -134,   221,  -134,   221,   221,  -134,   221,   221,  -134,
     221,   221,   221,   221,   221,  -134,   221,  -134,   199,    63,
     135,  -134,   -51,  -134,   136,    73,  -134,  -134,   199,   221,
    -134,  -134,   144,  -134,    21,  -134,   147,  -134,     6,   199,
    -134,   150,  -134,   199,  -134,  -134,  -134,   199,  -134,   199,
    -134,     9,  -134,   102,  -134,    69,    71,  -134,  -134,  -134,
     134,    40,    38,    45,   -41,   -41,   -13,   -13,    39,    39,
      39,    39,    39,  -134,   105,    96,  -134,    92,   140,    94,
      99,    93,  -134,  -134,   -34,   104,  -134,    97,   199,   108,
     109,  -134,    31,    30,    55,  -134,    81,  -134,  -134,  -134,
     111,  -134,    58,   117,  -134,  -134,    60,  -134,   199,  -134,
    -134,  -134,  -134,  -134,  -134,  -134,  -134,  -134,  -134,  -134,
    -134,  -134,  -134,  -134,  -134,   199,  -134,  -134,   199,  -134,
     199,   112,  -134,   140,  -134,  -134,    63,  -134,   199,   199,
    -134,  -134,   136,  -134,  -134,  -134,  -134,  -134,  -134,  -134,
    -134,   183,  -134,  -134,  -134,  -134,   102,  -134,  -134,    92,
    -134,  -134,   199,  -134,  -134,  -134,    93,  -134,   138,  -134,
      97,   111,  -134,  -134,  -134,  -134,   199,  -134,  -134,  -134,
    -134
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -134,   152,  -134,  -134,  -134,  -134,  -134,  -134,  -134,  -134,
    -134,  -112,  -134,   -83,  -134,  -134,  -134,  -134,   -94,  -134,
    -134,  -134,  -134,   169,   -38,  -134,  -134,  -134,   -29,    35,
    -134,  -105,   -81,   -28,    32,   -26,    -7,    -4,    26,  -134,
     184,   -10,  -134,  -134,    29,   110,    36,    -9,  -134,    46,
    -134,  -134,    -3,    24,   100,    47,    91,    50,    98,  -133,
     -71,  -126,   -65,   -84,   -21,  -134,  -134,  -134,    77,  -134,
      62,  -134,  -134,  -134,  -134,  -134,   173,  -134,     1,  -134,
      65,  -134,   -44,  -134,   -20,  -134,  -134,   -50,  -134,   -19,
    -134,   -11,  -134,   -42
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      63,    64,    62,    69,    68,    82,    83,    67,    85,   226,
     134,   135,    76,    77,    58,   148,    60,   149,    65,   268,
     137,   138,    67,   150,    70,    90,    86,    88,     1,     2,
       3,     4,     5,     6,     7,     8,     9,    10,    11,   269,
      78,   227,    12,    94,    13,    14,    15,    16,    17,    74,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      84,   246,   247,   194,   195,    28,     1,     2,     3,     4,
     248,   249,   196,   197,   140,    29,   141,   142,   143,   144,
      12,    96,    13,    14,    15,   113,   115,   130,   128,   181,
     146,    30,   132,    31,   183,    32,    26,    27,   158,   154,
     156,   157,    33,   159,   162,   166,   186,   161,   184,   168,
     173,   176,   178,    29,   250,   251,   252,   253,   254,   198,
     199,   200,   201,   202,   190,   203,   175,   164,   169,    30,
     171,    31,   179,    32,   177,   187,   208,   188,   212,   214,
      33,   204,   209,     1,     2,     3,     4,   223,   117,   218,
     225,   118,   119,   232,   220,   238,   222,    12,   255,    13,
      14,    15,   231,   257,   258,   266,   204,   263,   265,   272,
     234,   271,    90,    26,    27,   120,   121,   122,   123,   124,
     125,   275,   276,   281,   284,   292,   226,    95,   308,   300,
      29,   306,   302,    80,   283,   279,   309,   278,   280,   224,
     301,   261,     1,     2,     3,     4,    30,   237,    31,    81,
      32,   274,   277,   260,   240,   243,    12,    33,    13,    14,
      15,   192,   241,   185,     1,     2,     3,     4,   191,   213,
     193,   153,    26,    27,   286,   233,   242,   285,   289,   244,
      13,    14,    15,   245,   236,   303,   305,   296,   287,    29,
     304,   204,   294,   290,    26,    27,   261,     0,     0,   209,
       0,   297,   298,     0,     0,    30,     0,    31,     0,    32,
       0,    29,     0,     0,     0,     0,    33,     0,     0,     0,
       0,     0,     0,     0,     0,   290,     0,    30,     0,    31,
       0,    32,     0,     0,     0,     0,     0,     0,    33,   310,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110
};

static const yytype_int16 yycheck[] =
{
       7,     8,     6,    12,    11,    26,    27,    10,    29,     3,
      51,    52,    19,    20,     6,    66,     3,    68,    73,    53,
      33,    34,    25,    74,     3,    32,    30,    31,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    73,
       3,    35,    17,     0,    19,    20,    21,    22,    23,    17,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      28,   194,   195,   134,   135,    40,     3,     4,     5,     6,
     196,   197,   137,   138,    35,    50,    37,    38,    39,    40,
      17,    72,    19,    20,    21,    16,    15,    49,    48,    96,
      36,    66,    47,    68,    98,    70,    33,    34,    14,    66,
      73,    73,    77,    72,    74,    24,   115,    73,   112,    22,
      66,    69,    71,    50,   198,   199,   200,   201,   202,   140,
     141,   142,   143,   144,   127,   146,    67,    72,    72,    66,
      72,    68,    72,    70,    73,    14,    73,    17,     3,     3,
      77,   148,   149,     3,     4,     5,     6,     3,    14,    76,
       3,    17,    18,     3,   158,    53,   159,    17,    53,    19,
      20,    21,   169,    67,    72,    72,   173,    73,    69,    72,
     177,    67,   179,    33,    34,    41,    42,    43,    44,    45,
      46,    73,    73,    72,    67,    73,     3,    35,   300,   272,
      50,    53,   286,    24,   232,   224,   301,   223,   226,   164,
     281,   208,     3,     4,     5,     6,    66,   181,    68,    25,
      70,   218,   222,    73,   185,   191,    17,    77,    19,    20,
      21,   130,   186,   113,     3,     4,     5,     6,   128,   152,
     132,    58,    33,    34,   238,   173,   190,   236,   258,   192,
      19,    20,    21,   193,   179,   289,   296,   266,   255,    50,
     292,   258,   263,   260,    33,    34,   263,    -1,    -1,   266,
      -1,   268,   269,    -1,    -1,    66,    -1,    68,    -1,    70,
      -1,    50,    -1,    -1,    -1,    -1,    77,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   292,    -1,    66,    -1,    68,
      -1,    70,    -1,    -1,    -1,    -1,    -1,    -1,    77,   306,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    17,    19,    20,    21,    22,    23,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    40,    50,
      66,    68,    70,    77,    80,    81,    82,    83,    85,    95,
     104,   115,   116,   122,   124,   126,   127,   131,   133,   135,
     137,   139,   141,   143,   144,   146,   150,   154,     6,   155,
       3,    99,   116,   115,   115,    73,   119,   131,   115,   126,
       3,   106,   108,   113,   113,   101,   115,   115,     3,   102,
     102,   119,   143,   143,   113,   143,   116,   151,   116,   152,
     115,   153,   156,   159,     0,    80,    72,   117,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    96,    98,    16,   123,    15,   125,    14,    17,    18,
      41,    42,    43,    44,    45,    46,   128,   129,    48,   132,
      49,   134,    47,   136,    51,    52,   138,    33,    34,   140,
      35,    37,    38,    39,    40,   142,    36,   145,    66,    68,
      74,   147,   148,   155,    66,    87,    73,    73,    14,    72,
     120,    73,    74,   114,    72,   107,    24,   112,    22,    72,
     100,    72,   103,    66,    86,    67,    69,    73,    71,    72,
     157,   115,   118,   116,   116,   124,   126,    14,    17,   130,
     131,   133,   135,   137,   139,   139,   141,   141,   143,   143,
     143,   143,   143,   143,   115,   149,   160,   163,    73,   115,
     165,   168,     3,   147,     3,    88,    89,    92,    76,    84,
     116,   121,   131,     3,   108,     3,     3,    35,   105,   109,
     111,   115,     3,   149,   115,   158,   159,   117,    53,    97,
     123,   125,   128,   132,   134,   136,   138,   138,   140,   140,
     142,   142,   142,   142,   142,    53,   164,    67,    72,   161,
      73,   115,   170,    73,   169,    69,    72,   166,    53,    73,
      93,    67,    72,    90,   115,    73,    73,   120,   114,   107,
     112,    72,   110,   103,    67,   157,   116,   115,   162,   163,
     115,   172,    73,   171,   170,   167,   168,   115,   115,    91,
      92,   111,    97,   161,   172,   166,    53,    94,    90,   110,
     115
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
      
/* Line 1267 of yacc.c.  */
#line 1806 "y.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 557 "newparser.y"


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

