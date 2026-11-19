// SPDX-License-Identifier: GPL-2.0-or-later

use spirv::Op;

/// Naga can lower an explicit case sharing the default SPIR-V block as an
/// empty case that falls through to the next unrelated case. Remove that
/// redundant literal so the selector reaches the original default block.
pub fn lower(input: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
    const HEADER: usize = 5;
    let word_count = input.len() / 4;
    if word_count < HEADER || !input.len().is_multiple_of(4) {
        return Ok(None);
    }
    let word =
        |index: usize| u32::from_le_bytes(input[index * 4..index * 4 + 4].try_into().unwrap());
    if word(0) != spirv::MAGIC_NUMBER {
        return Ok(None);
    }

    let mut output = Vec::new();
    let mut copied_until = 0;
    let mut cursor = HEADER;
    while cursor < word_count {
        let instruction = word(cursor);
        let length = (instruction >> 16) as usize;
        if length == 0 || length > word_count - cursor {
            return Err("invalid SPIR-V instruction size");
        }
        let opcode = instruction & 0xffff;
        let end = cursor + length;
        if opcode == Op::TypeInt as u32 && length == 4 && word(cursor + 2) > 32 {
            // A wide switch selector can use two-word literals. Leave modules
            // with wide integers to Naga's normal import and validation path.
            return Ok(None);
        }
        if opcode == Op::Switch as u32 {
            if length < 3 || (length - 3) % 2 != 0 {
                return Err("invalid SPIR-V switch");
            }
            let default = word(cursor + 2);
            let cases = (cursor + 3..end).step_by(2);
            let aliases = cases
                .clone()
                .filter(|&index| word(index + 1) == default)
                .count();
            if aliases != 0 {
                output.extend_from_slice(&input[copied_until * 4..cursor * 4]);
                let header = (((length - 2 * aliases) as u32) << 16) | opcode;
                output.extend_from_slice(&header.to_le_bytes());
                output.extend_from_slice(&input[(cursor + 1) * 4..(cursor + 3) * 4]);
                for index in cases.filter(|&index| word(index + 1) != default) {
                    output.extend_from_slice(&input[index * 4..(index + 2) * 4]);
                }
                copied_until = end;
            }
        }
        cursor = end;
    }
    if copied_until == 0 {
        return Ok(None);
    }
    output.extend_from_slice(&input[copied_until * 4..]);
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_only_literals_that_share_the_default_block() {
        let header = [spirv::MAGIC_NUMBER, 0x10000, 0, 10, 0];
        let tail = [1 << 16 | Op::Return as u32];
        let switch = |cases: &[u32]| {
            header
                .into_iter()
                .chain([((cases.len() as u32 + 3) << 16) | Op::Switch as u32, 1, 2])
                .chain(cases.iter().copied())
                .chain(tail)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let input = switch(&[0, 2, 1, 3, 2, 4, 3, 3, 4, 2]);
        let expected = switch(&[1, 3, 2, 4, 3, 3]);
        assert_eq!(lower(&input), Ok(Some(expected.clone())));
        assert_eq!(lower(&expected), Ok(None));
    }
}
