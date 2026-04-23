// AUTO-GENERATED — do not edit by hand.
// Regenerate with scripts/build/generate_insert_patterns.py
// 45 patterns extracted.
//
// Data model (to be mirrored in insert.h):
//   Symbol{is_rule, value}
//   AltBranch{symbols: vector<Symbol>}
//   QuantifierSpec{min, max, alternatives: vector<AltBranch>}
//   MatchElement = variant<Symbol, AltBranch, QuantifierSpec>
//
// min=max=1 with multiple alternatives means 'pick one branch',
// used to represent unquantified alternations at a position.

#pragma once

#include "insert.h"

#include <climits>
#include <string>

namespace mlir_fuzzer {

inline InsertPatterns get_insert_patterns() {
  InsertPatterns patterns;

  // definition_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, INT_MAX, {AltBranch{{Symbol{true, "definition"}}}}}});
    p.child_rules.insert("definition");
    patterns["definition_list"] = std::move(p);
  }

  // dialect_attribute
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, "#"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, 1, {AltBranch{{Symbol{true, "opaque_dialect_item"}}}, AltBranch{{Symbol{true, "pretty_dialect_item"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, ":"}, Symbol{true, "mlir_type"}}}}}});
    p.child_rules.insert("mlir_type");
    p.child_rules.insert("opaque_dialect_item");
    p.child_rules.insert("pretty_dialect_item");
    patterns["dialect_attribute"] = std::move(p);
  }

  // dialect_type
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, "!"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, 1, {AltBranch{{Symbol{true, "opaque_dialect_item"}}}, AltBranch{{Symbol{true, "pretty_dialect_item"}}}}}});
    p.child_rules.insert("opaque_dialect_item");
    p.child_rules.insert("pretty_dialect_item");
    patterns["dialect_type"] = std::move(p);
  }

  // dim_id_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "("}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "BARE_ID"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "BARE_ID"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, ")"}});
    p.child_rules.insert("BARE_ID");
    patterns["dim_id_list"] = std::move(p);
  }

  // dim_use_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "("}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "ssa_use_list"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, ")"}});
    p.child_rules.insert("ssa_use_list");
    patterns["dim_use_list"] = std::move(p);
  }

  // function_result_list_no_parens
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "function_result"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "function_result"}}}}}});
    p.child_rules.insert("function_result");
    patterns["function_result_list_no_parens"] = std::move(p);
  }

  // generic_module
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "string_literal"}});
    p.match_pattern.push_back(MatchElement{Symbol{false, "("}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "argument_list"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, ")"}});
    p.match_pattern.push_back(MatchElement{Symbol{false, "("}});
    p.match_pattern.push_back(MatchElement{Symbol{true, "region"}});
    p.match_pattern.push_back(MatchElement{Symbol{false, ")"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "attribute_dict"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{true, "trailing_type"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "trailing_location"}}}}}});
    p.child_rules.insert("argument_list");
    p.child_rules.insert("attribute_dict");
    p.child_rules.insert("region");
    p.child_rules.insert("string_literal");
    p.child_rules.insert("trailing_location");
    p.child_rules.insert("trailing_type");
    patterns["generic_module"] = std::move(p);
  }

  // mlir_file
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, INT_MAX, {AltBranch{{Symbol{true, "definition_and_module_list"}}}}}});
    p.child_rules.insert("definition_and_module_list");
    patterns["mlir_file"] = std::move(p);
  }

  // module_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, INT_MAX, {AltBranch{{Symbol{true, "generic_module"}}}, AltBranch{{Symbol{true, "operation"}}}}}});
    p.child_rules.insert("generic_module");
    p.child_rules.insert("operation");
    patterns["module_list"] = std::move(p);
  }

  // op_result_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "op_result"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "op_result"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, "="}});
    p.child_rules.insert("op_result");
    patterns["op_result_list"] = std::move(p);
  }

  // operation_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, INT_MAX, {AltBranch{{Symbol{true, "operation"}}}}}});
    p.child_rules.insert("operation");
    patterns["operation_list"] = std::move(p);
  }

  // optional_arg_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "argument_list"}}}}}});
    p.child_rules.insert("argument_list");
    patterns["optional_arg_list"] = std::move(p);
  }

  // optional_attr_dict
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "attribute_dict"}}}}}});
    p.child_rules.insert("attribute_dict");
    patterns["optional_attr_dict"] = std::move(p);
  }

  // optional_block_arg_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "block_arg_list"}}}}}});
    p.child_rules.insert("block_arg_list");
    patterns["optional_block_arg_list"] = std::move(p);
  }

  // optional_block_label
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "block_label"}}}}}});
    p.child_rules.insert("block_label");
    patterns["optional_block_label"] = std::move(p);
  }

  // optional_fn_body
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "function_body"}}}}}});
    p.child_rules.insert("function_body");
    patterns["optional_fn_body"] = std::move(p);
  }

  // optional_fn_result_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, "->"}, Symbol{true, "function_result_list"}}}}}});
    p.child_rules.insert("function_result_list");
    patterns["optional_fn_result_list"] = std::move(p);
  }

  // optional_func_mod_attrs
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, "attributes"}, Symbol{true, "attribute_dict"}}}}}});
    p.child_rules.insert("attribute_dict");
    patterns["optional_func_mod_attrs"] = std::move(p);
  }

  // optional_int_literal
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, ":"}, Symbol{true, "integer_literal"}}}}}});
    p.child_rules.insert("integer_literal");
    patterns["optional_int_literal"] = std::move(p);
  }

  // optional_op_result_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "op_result_list"}}}}}});
    p.child_rules.insert("op_result_list");
    patterns["optional_op_result_list"] = std::move(p);
  }

  // optional_prop_dict
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "property_dict"}}}}}});
    p.child_rules.insert("property_dict");
    patterns["optional_prop_dict"] = std::move(p);
  }

  // optional_region_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "region_list"}}}}}});
    p.child_rules.insert("region_list");
    patterns["optional_region_list"] = std::move(p);
  }

  // optional_ssa_and_type_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "ssa_id_and_type_list"}}}}}});
    p.child_rules.insert("ssa_id_and_type_list");
    patterns["optional_ssa_and_type_list"] = std::move(p);
  }

  // optional_ssa_use_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "ssa_use_list"}}}}}});
    p.child_rules.insert("ssa_use_list");
    patterns["optional_ssa_use_list"] = std::move(p);
  }

  // optional_successor_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "successor_list"}}}}}});
    p.child_rules.insert("successor_list");
    patterns["optional_successor_list"] = std::move(p);
  }

  // optional_symbol_id_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "symbol_id_list"}}}}}});
    p.child_rules.insert("symbol_id_list");
    patterns["optional_symbol_id_list"] = std::move(p);
  }

  // optional_symbol_ref_id
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "symbol_ref_id"}}}}}});
    p.child_rules.insert("symbol_ref_id");
    patterns["optional_symbol_ref_id"] = std::move(p);
  }

  // optional_symbol_use_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "symbol_use_list"}}}}}});
    p.child_rules.insert("symbol_use_list");
    patterns["optional_symbol_use_list"] = std::move(p);
  }

  // optional_trailing_loc
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "trailing_location"}}}}}});
    p.child_rules.insert("trailing_location");
    patterns["optional_trailing_loc"] = std::move(p);
  }

  // optional_type
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, ":"}, Symbol{true, "mlir_type"}}}}}});
    p.child_rules.insert("mlir_type");
    patterns["optional_type"] = std::move(p);
  }

  // pretty_dialect_item
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "BARE_ID"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, "."}, Symbol{true, "BARE_ID"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "pretty_dialect_item_body"}}}}}});
    p.child_rules.insert("BARE_ID");
    p.child_rules.insert("pretty_dialect_item_body");
    patterns["pretty_dialect_item"] = std::move(p);
  }

  // pretty_dialect_item_contents
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, INT_MAX, {AltBranch{{Symbol{true, "pretty_dialect_bracketed"}}}, AltBranch{{Symbol{true, "pretty_dialect_item_other_content"}}}}}});
    p.child_rules.insert("pretty_dialect_bracketed");
    p.child_rules.insert("pretty_dialect_item_other_content");
    patterns["pretty_dialect_item_contents"] = std::move(p);
  }

  // region
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "{"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{true, "block"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, "}"}});
    p.child_rules.insert("block");
    patterns["region"] = std::move(p);
  }

  // region_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "("}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "region"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "region"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, ")"}});
    p.child_rules.insert("region");
    patterns["region_list"] = std::move(p);
  }

  // ssa_id
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "%"}});
    p.match_pattern.push_back(MatchElement{Symbol{true, "suffix_id"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{false, "#"}, Symbol{true, "DIGITS"}}}}}});
    p.child_rules.insert("DIGITS");
    p.child_rules.insert("suffix_id");
    patterns["ssa_id"] = std::move(p);
  }

  // ssa_id_and_type_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "ssa_id_and_type"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "ssa_id_and_type"}}}}}});
    p.child_rules.insert("ssa_id_and_type");
    patterns["ssa_id_and_type_list"] = std::move(p);
  }

  // ssa_id_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "ssa_id"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "ssa_id"}}}}}});
    p.child_rules.insert("ssa_id");
    patterns["ssa_id_list"] = std::move(p);
  }

  // ssa_use_and_type_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "ssa_use_and_type"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "ssa_use_and_type"}}}}}});
    p.child_rules.insert("ssa_use_and_type");
    patterns["ssa_use_and_type_list"] = std::move(p);
  }

  // ssa_use_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "ssa_use"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "ssa_use"}}}}}});
    p.child_rules.insert("ssa_use");
    patterns["ssa_use_list"] = std::move(p);
  }

  // successor_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "["}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "block_id"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "block_id"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, "]"}});
    p.child_rules.insert("block_id");
    patterns["successor_list"] = std::move(p);
  }

  // symbol_id_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "["}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "BARE_ID"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "BARE_ID"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, "]"}});
    p.child_rules.insert("BARE_ID");
    patterns["symbol_id_list"] = std::move(p);
  }

  // symbol_ref_attribute
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "symbol_ref_id"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, "::"}, Symbol{true, "symbol_ref_id"}}}}}});
    p.child_rules.insert("symbol_ref_id");
    patterns["symbol_ref_attribute"] = std::move(p);
  }

  // symbol_ref_id
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "@"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{1, 1, {AltBranch{{Symbol{true, "suffix_id"}}}, AltBranch{{Symbol{true, "string_literal"}}}}}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, "."}, Symbol{true, "suffix_id"}}}}}});
    p.child_rules.insert("string_literal");
    p.child_rules.insert("suffix_id");
    patterns["symbol_ref_id"] = std::move(p);
  }

  // symbol_use_list
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{false, "["}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, 1, {AltBranch{{Symbol{true, "ssa_use_list"}}}}}});
    p.match_pattern.push_back(MatchElement{Symbol{false, "]"}});
    p.child_rules.insert("ssa_use_list");
    patterns["symbol_use_list"] = std::move(p);
  }

  // type_list_no_parens
  {
    InsertPattern p;
    p.match_pattern.push_back(MatchElement{Symbol{true, "mlir_type"}});
    p.match_pattern.push_back(MatchElement{QuantifierSpec{0, INT_MAX, {AltBranch{{Symbol{false, ","}, Symbol{true, "mlir_type"}}}}}});
    p.child_rules.insert("mlir_type");
    patterns["type_list_no_parens"] = std::move(p);
  }

  return patterns;
}

} // namespace mlir_fuzzer
