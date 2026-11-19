// SPDX-License-Identifier: GPL-2.0-or-later

use naga::{Block, Module, Span, Statement};
use std::collections::HashMap;

pub fn lower(module: &mut Module) -> bool {
    let mut expanded = false;
    for (_, function) in module.functions.iter_mut() {
        lower_block(&mut function.body, &mut expanded);
    }
    for entry_point in &mut module.entry_points {
        lower_block(&mut entry_point.function.body, &mut expanded);
    }
    expanded
}

fn lower_block(block: &mut Block, expanded: &mut bool) {
    for statement in block.iter_mut() {
        match statement {
            Statement::Block(nested) => lower_block(nested, expanded),
            Statement::If { accept, reject, .. } => {
                lower_block(accept, expanded);
                lower_block(reject, expanded);
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                lower_block(body, expanded);
                lower_block(continuing, expanded);
            }
            Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    lower_block(&mut case.body, expanded);
                }
                for index in 0..cases.len() {
                    let (earlier, following) = cases.split_at_mut(index + 1);
                    let case = &mut earlier[index];
                    if !case.fall_through || case.body.is_empty() {
                        continue;
                    }
                    // WGSL can group empty case labels, but cannot fall through
                    // after executing statements. Inline the following bodies.
                    for next in following {
                        if !next.body.is_empty() {
                            case.body
                                .push(Statement::Block(next.body.clone()), Span::UNDEFINED);
                        }
                        if !next.fall_through {
                            break;
                        }
                    }
                    case.fall_through = false;
                    *expanded = true;
                }
            }
            _ => {}
        }
    }
}

/// Reused IR handles may cause the WGSL writer to emit synthetic declarations
/// like `let _e25 = _e25;` after the expanded case's lexical block has ended.
/// Reuse the actual initializer from the first occurrence in the same function.
pub fn repair_declarations(source: String) -> String {
    let mut declarations = HashMap::<&str, &str>::new();
    let mut output = String::with_capacity(source.len());
    for line in source.split_inclusive('\n') {
        if line.starts_with("fn ") {
            declarations.clear();
        }
        if let Some((name, initializer)) = generated_declaration(line.trim()) {
            if initializer == name {
                if let Some(first) = declarations.get(name) {
                    output.push_str(&line[..line.len() - line.trim_start().len()]);
                    output.push_str("let ");
                    output.push_str(name);
                    output.push_str(" = ");
                    output.push_str(first);
                    output.push_str(";\n");
                    continue;
                }
            } else {
                declarations.entry(name).or_insert(initializer);
            }
        }
        output.push_str(line);
    }
    output
}

fn generated_declaration(line: &str) -> Option<(&str, &str)> {
    let (name, initializer) = line.strip_prefix("let ")?.split_once(" = ")?;
    let index = name.strip_prefix("_e")?;
    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some((name, initializer.strip_suffix(';')?))
}
