grammar mlir;

start:
    mlir_file
;

mlir_file:
    definition_and_module_list+
;


// ---------------------------------------------------------------------- Structure of an MLIR
// parse-able string

definition_list:
    definition+
;

// Top-level modules in generic form now routinely carry a property dict
// (`"builtin.module"() <{sym_name = "x"}> ({...}) : () -> ()`) which
// `generic_module` can't express.  `generic_operation` already handles this
// identically, so accept it here too.  Cluster: `mismatched '<'` on the
// outer module.  (Q2)
module_list:
    (generic_module | operation)+
;

definition_and_module_list:
    definition_list
    | module_list
    | definition_list module_list
;



// Identifier syntax
suffix_id:
    DIGITS
    | BARE_ID
;


// ---------------------------------------------------------------------- Identifiers

ssa_id:
    '%' suffix_id ('#' DIGITS)?
;

// MLIR symbol references can contain dots: `@glob.private`, `@module.func`,
// `@acc.routine`.  Cluster `mismatched '.'` was overwhelmingly symbol refs
// whose dotted tail failed to parse.  (Q2 — opaque reference; mutator swaps
// symbol targets at the attribute level, not within names.)
symbol_ref_id:
    '@' (suffix_id | string_literal) ('.' suffix_id)*
;

block_id:
    '^' suffix_id
;

type_alias:
    '!' (string_literal | BARE_ID | (BARE_ID | '.')+)
;

map_or_set_id:
    '#' suffix_id
;

// `'loc'` is an ANTLR-implicit keyword (because `loc_attribute` / the old
// `trailing_location` use it as a literal) and therefore no longer matches
// BARE_ID.  `#loc = loc(...)` seeds (and `#loc1`, `#loc2`, etc. – those still
// match BARE_ID since the tail of the name isn't just the keyword).  Accept
// the literal `'loc'` here so top-level `#loc = ...` alias defs parse.  (Q2)
attribute_alias:
    '#' (string_literal | BARE_ID | 'loc')
;

ssa_id_list:
    ssa_id (',' ssa_id)*
;

// Uses of an SSA value, e.g., in an operand list to an operation.
ssa_use:
    ssa_id
    | constant_literal
;

ssa_use_list:
    ssa_use (',' ssa_use)*
;

// ---------------------------------------------------------------------- Types

// Dialect types - these can be opaque, pretty, or using custom dialects
opaque_dialect_item:
    BARE_ID '<' string_literal '>'
;

// Dialect names can chain multiple dots: `!gpu.async.token`,
// `!sparse_tensor.iter_space`, `!test.nested.inner.leaf`. The old rule capped
// at one dot, so the dialect-type alt consumed `a.b` and left `.rest` stuck.
// Cluster: tail of `mismatched '.'`.  (Q2 — nested dialect paths.)
pretty_dialect_item:
    BARE_ID ('.' BARE_ID)* pretty_dialect_item_body?
;

// Empty bodies like `#cuda_tile.optimization_hints<>` appear in the
// property-dictionary form emitted by --mlir-print-op-generic.
pretty_dialect_item_body:
    '<' '>'
    | '<' pretty_dialect_item_contents
    (
        ',' pretty_dialect_item_contents
    )* '>'
;

// A single "content" element between top-level commas of a pretty dialect
// body is a free mix of bracketed groups and bare atoms.  Scalable-vector
// types like `vector<[4]xf32>` produce `[4]` then `xf32`, and ArmSME tiles
// like `vector<[8]x[8]xi16>` interleave bracketed/atom pairs.  The old split
// rule could not express either. (Q2 — cluster: `extraneous 'xf32'`, `'xi1'`,
// `'xi8'`, scalable vector shapes.)
pretty_dialect_item_contents:
    (pretty_dialect_bracketed | pretty_dialect_item_other_content)+
;

pretty_dialect_bracketed:
    '(' (pretty_dialect_item_contents (',' pretty_dialect_item_contents)*)? ')'
    | '[' (pretty_dialect_item_contents (',' pretty_dialect_item_contents)*)? ']'
    | '{' (pretty_dialect_item_contents (',' pretty_dialect_item_contents)*)? '}'
    // LLVMIR debug forms nest `<...>` inside dialect bodies, e.g.
    // `#llvm.access_group<id = distinct[0]<>>`.  Adding angle-bracket as a
    // bracketed alternative lets the inner `<>` (or `<content>`) close before
    // the outer body's `>`.  (Q2)
    | '<' (pretty_dialect_item_contents (',' pretty_dialect_item_contents)*)? '>'
;

// Atoms of a single "content" between top-level body commas.
// - Use `non_function_type` instead of `mlir_type`: dropping the function_type
//   alt cuts the `->` lookahead ANTLR would otherwise perform at every body
//   token. Nested function types inside dialect bodies (e.g. `!llvm.func<()->()>`)
//   are rare; widen again if a seed needs them.
// - `,` is removed: the outer body splits content on `,`, so having it as an
//   inner atom makes every comma ambiguous.
// - `=` supports key/value pairs like `strides=[1]`, `sm_100 = {...}`,
//   `padding_value = zero`.
// - `->`, `+`, `-`, `/`, `|`, and the affine keywords `floordiv`, `ceildiv`,
//   `mod` are added for Q1 affine-expression coverage: `affine_map<(d0)->(d0+1)>`,
//   `affine_map<(d0,d1)->(d0 floordiv 2, d1 mod 3)>`, and for the OpenMP
//   `#omp<clause_map_flags to|from|implicit>` bit-flag form.  They live as
//   flat atom alternatives rather than a wrapper rule so the mutator's
//   flattened child sequence under `pretty_dialect_item_contents` is preserved.
// - `.` is added for version triplets and similar, e.g. `#spirv.vce<v1.3, ...>`
//   where `v1` / `3` are separate tokens joined by a literal period.
pretty_dialect_item_other_content:
    non_function_type
    | hashed_dialect_attribute
    | '*'
    | '?'
    | ':'
    | '='
    | '->'
    | '+'
    | '-'
    | '/'
    | '|'
    | '.'
    | 'floordiv'
    | 'ceildiv'
    | 'mod'
    | '>='
    | '<='
    | '=='
    | '!='
;

// A `#`-prefixed dialect-attribute reference used as an atom inside another
// dialect body, e.g. `memref<64xf16, #gpu.address_space<global>>` or
// `!sparse_tensor.iter_space<#sparse, lvls = 0 to 2>`.  Distinct from the
// top-level `dialect_attribute` rule (whose `#` is optional and therefore
// ambiguous with bare `non_function_type` atoms).  Cluster: `extraneous '#'`,
// `no viable alt at '<#'`.  (Q2 — opaque attribute pointer, mutator will
// substitute references at the attribute-dict level, not here.)
hashed_dialect_attribute:
    '#' (opaque_dialect_item | pretty_dialect_item)
;

dialect_type:
    '!'? (opaque_dialect_item | pretty_dialect_item)
;

// Order matters for ALL(*) speed. `type_alias` (`!foo.bar`) and `dialect_type`
// (`!foo.bar<...>`) both start with the same `!BARE_ID(.BARE_ID)*` prefix, so
// ANTLR always has to look ahead to see whether `<...>` follows. Putting
// `dialect_type` first means the common bodied form is the first candidate;
// type_alias only wins when there is no body or the alias name doesn't look
// like a pretty_dialect_item. `BARE_ID` as a separate alt was redundant with
// `dialect_type → pretty_dialect_item` (no prefix, no body) — removed.
non_function_type:
    dialect_type
    | type_alias
    | constant_literal
;

mlir_type:
    non_function_type
    | function_type
;

// Uses of types
type_list_no_parens:
    mlir_type (',' mlir_type)*
;

type_list_parens:
    ('(' ')')
    | ('(' type_list_no_parens ')')
;

ssa_use_and_type:
    ssa_use ':' mlir_type
;

ssa_use_and_type_list:
    ssa_use_and_type (',' ssa_use_and_type)*
;

// ---------------------------------------------------------------------- Attributes

// Simple attribute types
array_attribute:
    '[' (attribute_value ( ',' attribute_value)*)? ']'
;

bool_attribute:
    bool_literal
;

dictionary_attribute:
    '{' (attribute_entry ( ',' attribute_entry)*)? '}'
;

float_attribute:
    (FLOAT_LITERAL optional_type)
    | (HEXADECIMAL_LITERAL ':' mlir_type)
;

integer_attribute:
    posneg_integer_literal optional_type
;

string_attribute:
    string_literal optional_type
;

symbol_ref_attribute:
    (symbol_ref_id ( '::' symbol_ref_id)*)
;

type_attribute:
    mlir_type
;

// Standard attributes
standard_attribute:
    array_attribute
    | bool_attribute
    | dictionary_attribute
    | float_attribute
    | integer_attribute
    | string_attribute
    | symbol_ref_attribute
    | type_attribute
;

// `attribute_alias` (`#foo`) and `dialect_attribute` (`#foo.bar<body>`) share
// the same `#BARE_ID(.BARE_ID)?` prefix; put the more expressive
// `dialect_attribute` first so the common bodied form doesn't require a
// backtracked retry. Same rationale as `non_function_type` above.
attribute_value:
    dialect_attribute
    | attribute_alias
    | loc_attribute
    | standard_attribute
;

// Location attributes: `loc("file":L:C)`, `loc("name")`, `loc(fused<tag>[...])`,
// `loc(callsite(#loc at #loc1))`. Many seeds use `#loc = loc(...)` as a
// top-level attribute alias; `loc(...)` is neither a dialect attribute nor a
// standard attribute in the grammar. Keep it permissive with the existing
// balanced-content machinery.  (Q2 — attribute reference, not a mutation
// target for the fuzzer.)
loc_attribute:
    'loc' '(' (pretty_dialect_item_contents (',' pretty_dialect_item_contents)*)? ')'
;


dependent_attribute_entry:
    BARE_ID '=' attribute_value
;

dialect_attribute_entry:
    (BARE_ID '.' BARE_ID)
    | (BARE_ID '.' BARE_ID '=' attribute_value)
    | (string_literal '=' attribute_value)
;

// Dialect attributes
dialect_attribute:
    '#'? (opaque_dialect_item | pretty_dialect_item) (':' mlir_type)?
;

// Property dictionaries
property_dict:
    '<' attribute_dict '>'
;

// Attribute dictionaries
attribute_entry:
    dialect_attribute_entry
    | dependent_attribute_entry
    | BARE_ID
;

attribute_dict:
    ('{' '}')
    | ('{' attribute_entry (',' attribute_entry)* '}')
;

// ---------------------------------------------------------------------- Operations

// Types that appear after the operation, indicating return types
trailing_type:
    ':' function_type
;

// Only `->` appears in generic-form seeds; `to` / `into` are custom-op syntax.
// Listing them here made them ANTLR keywords, blocking legitimate uses of
// `to` / `into` as BARE_ID attribute values or keys (e.g.
// `capture_clause = (to)`, cluster `extraneous 'to'`).  (Q2)
function_type:
    function_type_list '->' function_type_list
;

// The paren-wrapped form must accept full `mlir_type` so nested function types
// parse: `func.constant` returning a function type appears as
// `() -> ((tensor<f32>) -> tensor<f32>)`.  The no-parens form keeps
// `non_function_type` only; widening it would introduce an unbounded `->`
// lookahead at every top-level type position.  (Q2)
function_type_list:
    '(' mlir_type? (',' mlir_type)* ')'
    | non_function_type? (',' non_function_type)*
;

// Operation results
op_result:
    ssa_id optional_int_literal
;

op_result_list:
    op_result (',' op_result)* '='
;

// Trailing location (for debug information)
location:
    string_literal ':' decimal_literal ':' decimal_literal
;

trailing_location:
    ('loc' '(' location ')')
;

// Undefined operations in all dialects
generic_operation:
    string_literal '(' optional_ssa_use_list ')' optional_successor_list optional_prop_dict
        optional_region_list optional_attr_dict trailing_type
;

custom_operation:
    BARE_ID '.' BARE_ID optional_ssa_use_list trailing_type
;

// Final operation definition NOTE: 'pymlir_dialect_ops' is defined externally by pyMLIR
// `generic_module` was one of the alternatives but it's a strict subset of
// `generic_operation` (no prop_dict, fewer optional bits) — any op matching
// `generic_module` also matches `generic_operation`. Keeping it created a
// duplicate reach-path that ALL(*) had to disambiguate at every string-literal
// op. Chief culprit of loop_split's slow parse (19 nested `"builtin.module"`).
// generic_module is still used by top-level `module_list` unchanged.
operation:
    optional_op_result_list
    (
        custom_operation
        | generic_operation
    ) optional_trailing_loc
;

// ---------------------------------------------------------------------- Blocks and regions

// Block arguments
ssa_id_and_type:
    ssa_id ':' mlir_type
;

ssa_id_and_type_list:
    ssa_id_and_type (',' ssa_id_and_type)*
;

block_arg_list:
    '(' optional_ssa_and_type_list ')'
;

operation_list:
    operation+
;

block_label:
    block_id optional_block_arg_list ':'
;

successor_list:
    '[' block_id? (',' block_id)* ']'
;

// Generic-form printer emits empty block bodies like `^bb0:` followed by
// nothing (when the entry block has block args but no ops), so allow a
// labeled block with zero operations.
block:
    block_label operation*
    | operation_list
;

region:
    '{' block* '}'
;

region_list:
    '(' region? (',' region)* ')'
;

// --------------------------------------------------------------------- ; Optional types ;
optional_symbol_ref_id:
    symbol_ref_id?
;

optional_func_mod_attrs:
    ('attributes' attribute_dict)?
;

optional_arg_list:
    argument_list?
;

optional_fn_result_list:
    ('->' function_result_list)?
;

optional_fn_body:
    function_body?
;

optional_symbol_id_list:
    symbol_id_list?
;

optional_type:
    (':' mlir_type)?
;

optional_int_literal:
    (':' integer_literal)?
;

optional_ssa_use_list:
    ssa_use_list?
;

optional_prop_dict:
    property_dict?
;

optional_attr_dict:
    attribute_dict?
;

optional_trailing_loc:
    trailing_location?
;

optional_op_result_list:
    op_result_list?
;

optional_ssa_and_type_list:
    ssa_id_and_type_list?
;

optional_block_arg_list:
    block_arg_list?
;

optional_block_label:
    block_label?
;

optional_symbol_use_list:
    symbol_use_list?
;

optional_successor_list:
    successor_list?
;

optional_region_list:
    region_list?
;

// ---------------------------------------------------------------------- Modules and functions

// Arguments
named_argument:
    ssa_id ':' mlir_type optional_attr_dict
;

argument_list:
    (named_argument ( ',' named_argument)*)
    | (mlir_type optional_attr_dict (',' mlir_type optional_attr_dict)*)
;

// Return values
function_result:
    mlir_type optional_attr_dict
;

function_result_list_no_parens:
    function_result (',' function_result)*
;

function_result_list_parens:
    ('(' ')')
    | ('(' function_result_list_no_parens ')')
;

function_result_list:
    function_result_list_parens
;

// Body
function_body:
    region
;

// Definition

generic_module:
    string_literal '(' argument_list? ')' '(' region ')' attribute_dict? trailing_type
        trailing_location?
;

// ---------------------------------------------------------------------- (semi-)affine expressions,
// maps, and integer sets

dim_id_list:
    '(' BARE_ID? (',' BARE_ID)* ')'
;

symbol_id_list:
    '[' BARE_ID? (',' BARE_ID)* ']'
;

dim_and_symbol_id_lists:
    dim_id_list optional_symbol_id_list
;

symbol_or_const:
    posneg_integer_literal
    | ssa_id
    | BARE_ID
;

dim_use_list:
    '(' ssa_use_list? ')'
;

symbol_use_list:
    '[' ssa_use_list? ']'
;

dim_and_symbol_use_list:
    dim_use_list optional_symbol_use_list
;

// ---------------------------------------------------------------------- General structure and
// top-level definitions

// Definitions of affine maps/integer sets/aliases are at the top of the file.
// The historical `'type'` literal made `type` an ANTLR-implicit keyword,
// which blocked every attribute-dict entry whose key is literally `type`
// (e.g. `{type = !emitc.array<1xf32>}`, common in EmitC / Cpp / SPIR-V /
// OpenACC seeds — cluster `mismatched 'type'`).  Generic-form seeds in this
// corpus don't use the `type` keyword at all, so the simpler equals-form
// covers them without introducing the keyword.  (Q2)
type_alias_def:
    type_alias '=' mlir_type
;

attribute_alias_def:
    attribute_alias '=' attribute_value
;

definition:
    type_alias_def
    | attribute_alias_def
;

// Tokens
ESCAPED_STRING:
    ([uUbB]? [rR]? | [rR]? [uUbB]?)
    (
        '\'' ('\\' ( ( [ \t]+ ( '\r'? '\n')?) | .) | ~[\\\r\n'])* '\''
        | '"' ('\\' (([ \t]+ ('\r'? '\n')?) | .) | ~[\\\r\n"])* '"'
        | '"""' ('\\' . | ~'\\')*? '"""'
        | '\'\'\'' ('\\' . | ~'\\')*? '\'\'\''
    )
;

// ---------------------------------------------------------------------- Literals



bool_literal:
    TRUE
    | FALSE
;

decimal_literal:
    DIGITS
;

HEXADECIMAL_LITERAL:
    '0x' [0-9a-fA-F]+
;

integer_literal:
    decimal_literal
    | HEXADECIMAL_LITERAL
;

negated_integer_literal:
    '-' integer_literal
;

posneg_integer_literal:
    integer_literal
    | negated_integer_literal
;

string_literal:
    ESCAPED_STRING
;

constant_literal:
    bool_literal
    | posneg_integer_literal
    | FLOAT_LITERAL
    | string_literal
;


FLOAT_LITERAL:
    [-+]? [0-9]+ [.][0-9]* ([eE][-+]? [0-9]+)?
;

DIGITS:
    DIGIT+
;

NONZERO_DIGIT:
    [1-9]
;

fragment DIGIT:
    [0-9]
;

fragment LETTER:
    [a-zA-Z]
;

TRUE:
    'true'
;

FALSE:
    'false'
;

fragment ID_CHARS:
    [$]
;

BARE_ID:
    (LETTER | '_') (LETTER | DIGIT | '_' | ID_CHARS)*
;

WS:
    [ \t\r\n]+ -> skip
;

// MLIR inherits LLVM-style line comments. Failing to skip them made every `//`
// in a FileCheck-stripped seed a stray `/` pair. Cluster: `at '/'` (Q2 — comments
// carry no grammar meaning).
LINE_COMMENT:
    '//' ~[\r\n]* -> skip
;

// Rare in seeds but cheap to add for symmetry.
BLOCK_COMMENT:
    '/*' .*? '*/' -> skip
;