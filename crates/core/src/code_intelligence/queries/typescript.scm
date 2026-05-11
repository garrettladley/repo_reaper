(function_declaration
  name: (identifier) @symbol.function)

(method_definition
  name: [(property_identifier) (private_property_identifier)] @symbol.function)

(class_declaration
  name: (type_identifier) @symbol.type)

(interface_declaration
  name: (type_identifier) @symbol.type)

(type_alias_declaration
  name: (type_identifier) @symbol.type)

(enum_declaration
  name: (identifier) @symbol.type)

(import_statement) @import

(comment) @comment

(string) @string

(template_string) @string
