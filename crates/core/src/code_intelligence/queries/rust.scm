(function_item
  name: (identifier) @symbol.function)

(function_signature_item
  name: (identifier) @symbol.function)

(struct_item
  name: (type_identifier) @symbol.type)

(enum_item
  name: (type_identifier) @symbol.type)

(trait_item
  name: (type_identifier) @symbol.trait)

(impl_item
  type: (_) @symbol.impl)

(use_declaration) @import

(line_comment) @comment

(block_comment) @comment

(string_literal) @string
