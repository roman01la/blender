// SPDX-License-Identifier: GPL-2.0-or-later

use naga::{
    BinaryOperator, Binding, Expression, Function, Handle, Literal, MathFunction, Module, Scalar,
    ScalarKind, ShaderStage, Span, Statement, Type, TypeInner,
};

/// The vertex fetch provides the packed word. Decode it at the entry point so
/// the imported shader keeps its original floating-point interface internally.
pub fn lower(module: &mut Module, locations: u32) -> Result<(), &'static str> {
    let entry = module
        .entry_points
        .iter_mut()
        .find(|entry| entry.stage == ShaderStage::Vertex)
        .ok_or("packed vertex attributes require a vertex entry point")?;
    let uint = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );

    let mut inner = std::mem::take(&mut entry.function);
    let wrapper = &mut entry.function;
    wrapper.arguments = inner.arguments.clone();
    wrapper.result = inner.result.clone();
    inner.name = Some("blender_unpacked_vertex".to_owned());
    if let Some(result) = &mut inner.result {
        result.binding = None;
    }

    let mut arguments = Vec::with_capacity(wrapper.arguments.len());
    let mut unpack = Vec::new();
    for (i, argument) in wrapper.arguments.iter_mut().enumerate() {
        let value = wrapper
            .expressions
            .append(Expression::FunctionArgument(i as u32), Span::UNDEFINED);
        if let Some(Binding::Location { location, .. }) = inner.arguments[i].binding.take() {
            if location < 32 && locations & (1 << location) != 0 {
                let ty = argument.ty;
                let lanes = match module.types[ty].inner {
                    TypeInner::Scalar(Scalar::F32) => 1,
                    TypeInner::Vector {
                        size,
                        scalar: Scalar::F32,
                    } => size as usize,
                    _ => return Err("packed normals require a floating-point vertex input"),
                };
                argument.ty = uint;
                unpack.push((i, value, ty, lanes));
            }
        }
        arguments.push(value);
    }

    for (i, value, ty, lanes) in unpack {
        arguments[i] = unpack_value(wrapper, value, ty, lanes);
    }
    let function = module.functions.append(inner, Span::UNDEFINED);
    let result = wrapper
        .result
        .is_some()
        .then(|| append(wrapper, Expression::CallResult(function)));
    wrapper.body.push(
        Statement::Call {
            function,
            arguments,
            result,
        },
        Span::UNDEFINED,
    );
    wrapper
        .body
        .push(Statement::Return { value: result }, Span::UNDEFINED);
    Ok(())
}

fn append(function: &mut Function, expr: Expression) -> Handle<Expression> {
    function.expressions.append(expr, Span::UNDEFINED)
}

fn unpack_value(
    function: &mut Function,
    value: Handle<Expression>,
    ty: Handle<Type>,
    lanes: usize,
) -> Handle<Expression> {
    let minus_one = append(function, Expression::Literal(Literal::F32(-1.0)));
    let params: Vec<_> = (0..lanes)
        .map(|i| {
            let offset = append(function, Expression::Literal(Literal::U32(i as u32 * 10)));
            let width = append(
                function,
                Expression::Literal(Literal::U32(if i == 3 { 2 } else { 10 })),
            );
            (offset, width)
        })
        .collect();
    let inv_511 = append(function, Expression::Literal(Literal::F32(1.0 / 511.0)));
    let mut emit = naga::proc::Emitter::default();
    emit.start(&function.expressions);
    let word = append(
        function,
        Expression::As {
            expr: value,
            kind: ScalarKind::Sint,
            convert: None,
        },
    );
    let components: Vec<_> = params
        .into_iter()
        .enumerate()
        .map(|(i, (offset, width))| {
            let extracted = append(
                function,
                Expression::Math {
                    fun: MathFunction::ExtractBits,
                    arg: word,
                    arg1: Some(offset),
                    arg2: Some(width),
                    arg3: None,
                },
            );
            let converted = append(
                function,
                Expression::As {
                    expr: extracted,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                },
            );
            let normalized = if i == 3 {
                converted
            } else {
                append(
                    function,
                    Expression::Binary {
                        op: BinaryOperator::Multiply,
                        left: converted,
                        right: inv_511,
                    },
                )
            };
            append(
                function,
                Expression::Math {
                    fun: MathFunction::Max,
                    arg: normalized,
                    arg1: Some(minus_one),
                    arg2: None,
                    arg3: None,
                },
            )
        })
        .collect();
    let result = if lanes == 1 {
        components[0]
    } else {
        append(function, Expression::Compose { ty, components })
    };
    function.body.extend(emit.finish(&function.expressions));
    result
}
