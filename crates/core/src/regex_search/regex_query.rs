use std::collections::BTreeSet;

use super::{Trigram, trigrams};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexCandidatePlan {
    All,
    And(Vec<Trigram>),
    Or(Vec<RegexCandidatePlan>),
}

impl RegexCandidatePlan {
    pub fn for_pattern(pattern: &str) -> Self {
        if has_case_insensitive_flag(pattern) {
            return Self::All;
        }

        let branches = split_top_level_alternation(pattern);
        if branches.len() == 1 {
            plan_for_concatenation(branches[0])
        } else {
            Self::Or(
                branches
                    .into_iter()
                    .map(plan_for_concatenation)
                    .collect::<Vec<_>>(),
            )
        }
    }
}

fn plan_for_concatenation(pattern: &str) -> RegexCandidatePlan {
    let mut analyzer = ConcatenationAnalyzer::default();
    analyzer.analyze(pattern);
    let trigrams = analyzer.required_trigrams();

    if trigrams.is_empty() {
        RegexCandidatePlan::All
    } else {
        RegexCandidatePlan::And(trigrams)
    }
}

#[derive(Debug, Default)]
struct ConcatenationAnalyzer {
    literals: Vec<String>,
    current_literal: String,
    last_atom_was_literal: bool,
}

impl ConcatenationAnalyzer {
    fn analyze(&mut self, pattern: &str) {
        let chars = pattern.chars().collect::<Vec<_>>();
        let mut index = 0;
        while index < chars.len() {
            match chars[index] {
                '\\' => {
                    let Some((literal, next_index)) = escaped_literal(&chars, index) else {
                        self.finish_literal();
                        self.last_atom_was_literal = false;
                        index += 2;
                        continue;
                    };
                    self.current_literal.push(literal);
                    self.last_atom_was_literal = true;
                    index = next_index;
                }
                '[' => {
                    let (literal, next_index) = character_class_literal(&chars, index);
                    if let Some(literal) = literal {
                        self.current_literal.push(literal);
                        self.last_atom_was_literal = true;
                    } else {
                        self.finish_literal();
                        self.last_atom_was_literal = false;
                    }
                    index = next_index;
                }
                '(' => {
                    self.finish_literal();
                    self.last_atom_was_literal = false;
                    index = skip_group(&chars, index);
                }
                '.' | '^' | '$' => {
                    self.finish_literal();
                    self.last_atom_was_literal = false;
                    index += 1;
                }
                '*' | '+' | '?' => {
                    if self.last_atom_was_literal {
                        self.remove_last_literal_char();
                    }
                    self.last_atom_was_literal = false;
                    index += 1;
                }
                '{' => {
                    if self.last_atom_was_literal {
                        self.remove_last_literal_char();
                    }
                    self.last_atom_was_literal = false;
                    index = skip_counted_repetition(&chars, index);
                }
                ')' | ']' => {
                    self.finish_literal();
                    self.last_atom_was_literal = false;
                    index += 1;
                }
                literal => {
                    self.current_literal.push(literal);
                    self.last_atom_was_literal = true;
                    index += 1;
                }
            }
        }

        self.finish_literal();
    }

    fn required_trigrams(&self) -> Vec<Trigram> {
        let mut seen = BTreeSet::new();
        for literal in &self.literals {
            seen.extend(trigrams(literal));
        }
        seen.into_iter().collect()
    }

    fn finish_literal(&mut self) {
        if !self.current_literal.is_empty() {
            self.literals
                .push(std::mem::take(&mut self.current_literal));
        }
    }

    fn remove_last_literal_char(&mut self) {
        self.current_literal.pop();
        if self.current_literal.is_empty() {
            self.literals.pop();
        }
    }
}

fn has_case_insensitive_flag(pattern: &str) -> bool {
    pattern.contains("(?i") || pattern.contains("(?-i")
}

fn split_top_level_alternation(pattern: &str) -> Vec<&str> {
    let chars = pattern.char_indices().collect::<Vec<_>>();
    let mut branches = Vec::new();
    let mut branch_start = 0;
    let mut class_depth = 0;
    let mut group_depth = 0;
    let mut index = 0;

    while index < chars.len() {
        let (byte_index, char_) = chars[index];
        match char_ {
            '\\' => index += 2,
            '[' if group_depth == 0 => {
                class_depth += 1;
                index += 1;
            }
            ']' if group_depth == 0 && class_depth > 0 => {
                class_depth -= 1;
                index += 1;
            }
            '(' if class_depth == 0 => {
                group_depth += 1;
                index += 1;
            }
            ')' if class_depth == 0 && group_depth > 0 => {
                group_depth -= 1;
                index += 1;
            }
            '|' if class_depth == 0 && group_depth == 0 => {
                branches.push(&pattern[branch_start..byte_index]);
                branch_start = byte_index + char_.len_utf8();
                index += 1;
            }
            _ => index += 1,
        }
    }

    branches.push(&pattern[branch_start..]);
    branches
}

fn escaped_literal(chars: &[char], index: usize) -> Option<(char, usize)> {
    let escaped = *chars.get(index + 1)?;
    match escaped {
        'n' | 'r' | 't' | 'd' | 'D' | 's' | 'S' | 'w' | 'W' | 'b' | 'B' | 'A' | 'z' | 'Z' | 'p'
        | 'P' | 'x' | 'u' => None,
        literal => Some((literal, index + 2)),
    }
}

fn character_class_literal(chars: &[char], start: usize) -> (Option<char>, usize) {
    let mut index = start + 1;
    let mut class_chars = Vec::new();
    let mut negated = false;

    if chars.get(index) == Some(&'^') {
        negated = true;
        index += 1;
    }

    while index < chars.len() {
        match chars[index] {
            '\\' => {
                let Some((literal, next_index)) = escaped_literal(chars, index) else {
                    return (None, skip_character_class(chars, start));
                };
                class_chars.push(literal);
                index = next_index;
            }
            ']' => {
                if !negated && class_chars.len() == 1 {
                    return (class_chars.first().copied(), index + 1);
                }
                return (None, index + 1);
            }
            '-' => return (None, skip_character_class(chars, start)),
            literal => {
                class_chars.push(literal);
                index += 1;
            }
        }
    }

    (None, chars.len())
}

fn skip_character_class(chars: &[char], start: usize) -> usize {
    let mut index = start + 1;
    while index < chars.len() {
        match chars[index] {
            '\\' => index += 2,
            ']' => return index + 1,
            _ => index += 1,
        }
    }
    chars.len()
}

fn skip_group(chars: &[char], start: usize) -> usize {
    let mut index = start + 1;
    let mut depth = 1;

    while index < chars.len() {
        match chars[index] {
            '\\' => index += 2,
            '[' => index = skip_character_class(chars, index),
            '(' => {
                depth += 1;
                index += 1;
            }
            ')' => {
                depth -= 1;
                index += 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => index += 1,
        }
    }

    chars.len()
}

fn skip_counted_repetition(chars: &[char], start: usize) -> usize {
    let mut index = start + 1;
    while index < chars.len() {
        if chars[index] == '}' {
            return index + 1;
        }
        index += 1;
    }
    chars.len()
}
