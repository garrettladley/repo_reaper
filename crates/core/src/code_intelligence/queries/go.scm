(function_declaration
  name: (identifier) @symbol.function)

(method_declaration
  name: (field_identifier) @symbol.function)

(type_declaration
  (type_spec
    name: (type_identifier) @symbol.type))

(import_declaration) @import

(comment) @comment

(interpreted_string_literal) @string

(raw_string_literal) @string
