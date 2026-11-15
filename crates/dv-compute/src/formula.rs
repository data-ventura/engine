//! Formulas: the formula bar's language, compiled to DataFusion expressions with units checked.
//!
//! A formula is checked against a dataset before it runs. Field names must exist, units must
//! agree, and functions must get the right arguments; each error points at the part of the
//! formula that caused it.
//!
//! **Units.** `+`, `-`, comparisons and `if` branches need convertible units, and the right side
//! is converted to the left side's unit (`height_ft + height_m` is in feet). `*`, `/` and `^`
//! combine units. A field whose unit couldn't be read blocks arithmetic until it's assigned one.
//! Numbers and fields without units fit any unit. `convert(x, "unit")` converts explicitly.
//!
//! **Numbers.** `/` always divides as floating point, so `5 / 2` is 2.5 even for integer fields.
//!
//! **Functions.**
//! - Row by row: `abs sqrt exp ln log10 round floor ceil`, `min(a, b)` and `max(a, b)`,
//!   `coalesce`, `if(cond, then, else)`, `is_null`, `year month day hour minute weekday`,
//!   `lower upper length contains`, `convert`.
//! - Over the whole column, repeated on every row: `mean sum min max count std var`. So
//!   `x - mean(x)` is the anomaly.
//! - In order (by the context's order field, usually time): `lag(x, n)`, `lead(x, n)`,
//!   `diff(x, n)`, `cumsum(x)`, `rolling_mean(x, n)`, `rolling_sum`, `rolling_min`,
//!   `rolling_max`. Rolling windows with fewer than `n` non-null values are null, and `cumsum`
//!   is null where `x` is null, as in polars and pandas.
//! - `rank(x)`: 1 for the smallest value, ties sharing the lowest rank (SQL `RANK`).

use arrow::datatypes::DataType;
use datafusion::common::ScalarValue;
use datafusion::functions::core::expr_fn::{coalesce, greatest, least};
use datafusion::functions::datetime::expr_fn::date_part;
use datafusion::functions::math::expr_fn as math;
use datafusion::functions::string::expr_fn::{contains, lower, upper};
use datafusion::functions::unicode::expr_fn::character_length;
use datafusion::functions_aggregate::average::avg_udaf;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::functions_aggregate::min_max::{max_udaf, min_udaf};
use datafusion::functions_aggregate::stddev::stddev_udaf;
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::functions_aggregate::variance::var_samp_udaf;
use datafusion::functions_window::expr_fn::{lag, lead, rank};
use datafusion::logical_expr::expr::WindowFunction;
use datafusion::logical_expr::{
    Expr, ExprFunctionExt, SortExpr, WindowFrame, WindowFrameBound, WindowFrameUnits,
    WindowFunctionDefinition, cast, lit, when,
};
use datafusion::prelude::ident;
use dv_model::field::Dataset;
use dv_model::units::Unit;

use crate::parse::{Ast, BinOp, FormulaError, Node, Span, parse};

/// What a formula is compiled against.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    pub dataset: &'a Dataset,
    /// The field that orders rows for `lag`, `cumsum`, rolling windows and the like.
    pub order_by: Option<&'a str>,
}

impl<'a> Context<'a> {
    /// A context ordered by the dataset's first time field, if it has one.
    pub fn new(dataset: &'a Dataset) -> Self {
        let order_by = dataset
            .with_role(dv_model::field::Role::Time)
            .next()
            .map(|f| f.name.as_str());
        Self { dataset, order_by }
    }
}

/// A compiled formula.
#[derive(Debug, Clone)]
pub struct Formula {
    pub expr: Expr,
    /// The result's unit, if it has one.
    pub unit: Option<Unit>,
    /// Fields the formula reads.
    pub fields: Vec<String>,
}

/// Parses and checks a formula, and compiles it to a DataFusion expression.
pub fn compile(src: &str, ctx: &Context<'_>) -> Result<Formula, FormulaError> {
    let node = parse(src)?;
    let mut c = Compiler {
        ctx,
        fields: Vec::new(),
    };
    let v = c.node(&node)?;
    Ok(Formula {
        expr: v.expr,
        unit: match v.unit {
            U::Known(u) => Some(u),
            _ => None,
        },
        fields: c.fields,
    })
}

/// A value's unit while compiling.
#[derive(Debug, Clone)]
enum U {
    /// No unit: a number, text, or a field without one. Fits any unit.
    None,
    Known(Unit),
    /// The field's unit text couldn't be read.
    Unknown(String),
}

struct V {
    expr: Expr,
    unit: U,
}

impl V {
    fn plain(expr: Expr) -> Self {
        V {
            expr,
            unit: U::None,
        }
    }
}

struct Compiler<'a, 'b> {
    ctx: &'b Context<'a>,
    fields: Vec<String>,
}

fn f64(e: Expr) -> Expr {
    cast(e, DataType::Float64)
}

fn err<T>(msg: impl Into<String>, span: Span) -> Result<T, FormulaError> {
    Err(FormulaError::new(msg, span))
}

impl Compiler<'_, '_> {
    fn node(&mut self, n: &Node) -> Result<V, FormulaError> {
        match &n.ast {
            Ast::Number(x) => Ok(V::plain(lit(*x))),
            Ast::Str(s) => Ok(V::plain(lit(s.clone()))),
            Ast::Bool(b) => Ok(V::plain(lit(*b))),
            Ast::Field(name) => self.field(name, n.span),
            Ast::Neg(a) => {
                let v = self.node(a)?;
                Ok(V {
                    expr: -v.expr,
                    unit: v.unit,
                })
            }
            Ast::Not(a) => Ok(V::plain(!self.node(a)?.expr)),
            Ast::Binary(op, a, b) => self.binary(*op, a, b, n.span),
            Ast::Call(name, args) => self.call(name, args, n.span),
        }
    }

    fn field(&mut self, name: &str, span: Span) -> Result<V, FormulaError> {
        let Some(f) = self.ctx.dataset.field(name) else {
            let hint = closest(
                name,
                self.ctx.dataset.fields.iter().map(|f| f.name.as_str()),
            )
            .map(|c| format!("; did you mean {c:?}?"))
            .unwrap_or_default();
            return err(format!("no field {name:?}{hint}"), span);
        };
        if !self.fields.iter().any(|x| x == name) {
            self.fields.push(name.to_string());
        }
        let unit = match (&f.unit, &f.unit_text) {
            (Some(u), _) => U::Known(u.clone()),
            (None, Some(t)) => U::Unknown(t.clone()),
            (None, None) => U::None,
        };
        Ok(V {
            expr: ident(name),
            unit,
        })
    }

    /// Fails if a value's unit is unreadable; arithmetic on it is blocked.
    fn known(&self, v: &V, span: Span) -> Result<(), FormulaError> {
        match &v.unit {
            U::Unknown(t) => err(
                format!("the unit {t:?} isn't recognized; assign this field a unit first"),
                span,
            ),
            _ => Ok(()),
        }
    }

    /// Brings `b` into `a`'s unit, for addition, comparison and the like.
    fn align(&self, a: &V, b: V, op: &str, span: Span) -> Result<(U, Expr), FormulaError> {
        Ok(match (&a.unit, b.unit) {
            (U::Known(ua), U::Known(ub)) => match ub.converter(ua) {
                Some(c) if c.is_identity() => (a.unit.clone(), b.expr),
                Some(c) => (a.unit.clone(), f64(b.expr) * lit(c.scale) + lit(c.offset)),
                None => {
                    return err(
                        format!("can't {op} {ub} and {ua}: the units don't convert"),
                        span,
                    );
                }
            },
            (U::Known(_), U::None) => (a.unit.clone(), b.expr),
            (U::None, u @ U::Known(_)) => (u, b.expr),
            _ => (U::None, b.expr),
        })
    }

    fn binary(&mut self, op: BinOp, a: &Node, b: &Node, span: Span) -> Result<V, FormulaError> {
        let va = self.node(a)?;
        let vb = self.node(b)?;
        use BinOp::*;
        if matches!(op, And | Or) {
            let e = if op == And {
                va.expr.and(vb.expr)
            } else {
                va.expr.or(vb.expr)
            };
            return Ok(V::plain(e));
        }
        self.known(&va, a.span)?;
        self.known(&vb, b.span)?;
        let verb = match op {
            Add => "add",
            Sub => "subtract",
            _ => "compare",
        };
        Ok(match op {
            Add | Sub | Eq | Ne | Lt | Le | Gt | Ge => {
                let left = va.expr.clone();
                let (unit, right) = self.align(&va, vb, verb, span)?;
                let e = match op {
                    Add => left + right,
                    Sub => left - right,
                    Eq => left.eq(right),
                    Ne => left.not_eq(right),
                    Lt => left.lt(right),
                    Le => left.lt_eq(right),
                    Gt => left.gt(right),
                    _ => left.gt_eq(right),
                };
                let unit = if matches!(op, Add | Sub) {
                    unit
                } else {
                    U::None
                };
                V { expr: e, unit }
            }
            Mul | Div => {
                let unit = match (&va.unit, &vb.unit) {
                    (U::Known(x), U::Known(y)) => {
                        U::Known(if op == Mul { x.mul(y) } else { x.div(y) })
                    }
                    (U::Known(x), U::None) => U::Known(x.clone()),
                    (U::None, U::Known(y)) if op == Mul => U::Known(y.clone()),
                    (U::None, U::Known(y)) => U::Known(y.powi(-1)),
                    _ => U::None,
                };
                let e = if op == Mul {
                    va.expr * vb.expr
                } else {
                    f64(va.expr) / f64(vb.expr)
                };
                V { expr: e, unit }
            }
            Mod => V {
                expr: va.expr % vb.expr,
                unit: va.unit,
            },
            Pow => {
                let unit = match (&va.unit, &b.ast) {
                    (U::Known(u), Ast::Number(n)) if n.fract() == 0.0 => {
                        U::Known(u.powi(*n as i32))
                    }
                    (U::Known(u), _) if !u.is_dimensionless() => {
                        return err(
                            "a value with a unit can only be raised to a whole number",
                            b.span,
                        );
                    }
                    _ => U::None,
                };
                V {
                    expr: math::power(f64(va.expr), f64(vb.expr)),
                    unit,
                }
            }
            And | Or => unreachable!("handled above"),
        })
    }

    fn args(
        &mut self,
        name: &str,
        args: &[Node],
        n: std::ops::RangeInclusive<usize>,
        span: Span,
    ) -> Result<Vec<V>, FormulaError> {
        if !n.contains(&args.len()) {
            let want = if n.start() == n.end() {
                format!("{}", n.start())
            } else {
                format!("{} to {}", n.start(), n.end())
            };
            return err(
                format!("{name}() takes {want} arguments, not {}", args.len()),
                span,
            );
        }
        args.iter().map(|a| self.node(a)).collect()
    }

    fn int_arg(&self, name: &str, a: &Node) -> Result<i64, FormulaError> {
        match a.ast {
            Ast::Number(x) if x.fract() == 0.0 && x >= 0.0 => Ok(x as i64),
            _ => err(format!("{name}() needs a whole number here"), a.span),
        }
    }

    fn order(&self, name: &str, span: Span) -> Result<Vec<SortExpr>, FormulaError> {
        match self.ctx.order_by {
            Some(f) => Ok(vec![ident(f).sort(true, true)]),
            None => err(
                format!("{name}() needs an order; this dataset has no time field to order by"),
                span,
            ),
        }
    }

    fn call(&mut self, name: &str, args: &[Node], span: Span) -> Result<V, FormulaError> {
        let same_unit = |v: V, e: Expr| V {
            expr: e,
            unit: v.unit,
        };
        match name {
            "abs" | "round" | "floor" | "ceil" => {
                let mut v = self.args(
                    name,
                    args,
                    if name == "round" { 1..=2 } else { 1..=1 },
                    span,
                )?;
                let x = v.remove(0);
                let e = match name {
                    "abs" => math::abs(x.expr.clone()),
                    "floor" => math::floor(f64(x.expr.clone())),
                    "ceil" => math::ceil(f64(x.expr.clone())),
                    _ => {
                        let digits = match args.get(1) {
                            Some(a) => self.int_arg(name, a)?,
                            None => 0,
                        };
                        math::round(vec![f64(x.expr.clone()), lit(digits)])
                    }
                };
                Ok(same_unit(x, e))
            }
            "sqrt" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                self.known(&x, args[0].span)?;
                let unit = match &x.unit {
                    U::Known(u) => match u.sqrt() {
                        Some(r) => U::Known(r),
                        None => return err(format!("sqrt() of {u} has no unit"), args[0].span),
                    },
                    _ => U::None,
                };
                Ok(V {
                    expr: math::sqrt(f64(x.expr)),
                    unit,
                })
            }
            "exp" | "ln" | "log10" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                if let U::Known(u) = &x.unit
                    && !u.is_dimensionless()
                {
                    return err(
                        format!("{name}() needs a value without a unit, not {u}"),
                        args[0].span,
                    );
                }
                self.known(&x, args[0].span)?;
                let e = f64(x.expr);
                Ok(V::plain(match name {
                    "exp" => math::exp(e),
                    "ln" => math::ln(e),
                    _ => math::log10(e),
                }))
            }
            "min" | "max" if args.len() == 2 => {
                let mut v = self.args(name, args, 2..=2, span)?;
                let b = v.pop().expect("two");
                let a = v.pop().expect("two");
                self.known(&a, args[0].span)?;
                self.known(&b, args[1].span)?;
                let left = a.expr.clone();
                let (unit, right) = self.align(&a, b, "compare", span)?;
                let e = if name == "min" {
                    least(vec![left, right])
                } else {
                    greatest(vec![left, right])
                };
                Ok(V { expr: e, unit })
            }
            "coalesce" => {
                let v = self.args(name, args, 1..=usize::MAX, span)?;
                let mut it = v.into_iter();
                let first = it.next().expect("at least one");
                let mut exprs = vec![first.expr.clone()];
                let mut unit = first.unit.clone();
                for (v, a) in it.zip(&args[1..]) {
                    self.known(&v, a.span)?;
                    let base = V {
                        expr: lit(0),
                        unit: unit.clone(),
                    };
                    let (u, e) = self.align(&base, v, "combine", a.span)?;
                    unit = u;
                    exprs.push(e);
                }
                Ok(V {
                    expr: coalesce(exprs),
                    unit,
                })
            }
            "if" => {
                let mut v = self.args(name, args, 3..=3, span)?;
                let otherwise = v.pop().expect("three");
                let then = v.pop().expect("three");
                let cond = v.pop().expect("three");
                let then_expr = then.expr.clone();
                let (unit, else_expr) = self.align(&then, otherwise, "combine", args[2].span)?;
                let e = when(cond.expr, then_expr)
                    .otherwise(else_expr)
                    .map_err(|e| FormulaError::new(e.to_string(), span))?;
                Ok(V { expr: e, unit })
            }
            "is_null" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                Ok(V::plain(x.expr.is_null()))
            }
            "year" | "month" | "day" | "hour" | "minute" | "weekday" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                let part = if name == "weekday" { "dow" } else { name };
                Ok(V::plain(cast(
                    date_part(lit(part), x.expr),
                    DataType::Int64,
                )))
            }
            "lower" | "upper" | "length" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                Ok(V::plain(match name {
                    "lower" => lower(x.expr),
                    "upper" => upper(x.expr),
                    _ => character_length(x.expr),
                }))
            }
            "contains" => {
                let mut v = self.args(name, args, 2..=2, span)?;
                let needle = v.pop().expect("two");
                let hay = v.pop().expect("two");
                Ok(V::plain(contains(hay.expr, needle.expr)))
            }
            "convert" => {
                let x = self.args(name, args, 2..=2, span)?.remove(0);
                let Ast::Str(target) = &args[1].ast else {
                    return err(
                        "convert() needs a unit in quotes, such as \"km\"",
                        args[1].span,
                    );
                };
                let to = Unit::parse(target)
                    .map_err(|e| FormulaError::new(e.to_string(), args[1].span))?;
                self.known(&x, args[0].span)?;
                let U::Known(from) = &x.unit else {
                    return err(
                        "this value has no unit to convert from; assign one first",
                        args[0].span,
                    );
                };
                let Some(c) = from.converter(&to) else {
                    return err(format!("can't convert {from} to {to}"), span);
                };
                Ok(V {
                    expr: f64(x.expr) * lit(c.scale) + lit(c.offset),
                    unit: U::Known(to),
                })
            }
            "mean" | "sum" | "min" | "max" | "count" | "std" | "var" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                self.known(&x, args[0].span)?;
                let udaf = match name {
                    "mean" => avg_udaf(),
                    "sum" => sum_udaf(),
                    "min" => min_udaf(),
                    "max" => max_udaf(),
                    "count" => count_udaf(),
                    "std" => stddev_udaf(),
                    _ => var_samp_udaf(),
                };
                let arg = if matches!(name, "mean" | "std" | "var") {
                    f64(x.expr)
                } else {
                    x.expr
                };
                let e = window(udaf, arg, vec![], None, span)?;
                let unit = match (name, x.unit) {
                    ("count", _) => U::None,
                    ("var", U::Known(u)) => U::Known(u.powi(2)),
                    (_, u) => u,
                };
                Ok(V { expr: e, unit })
            }
            "lag" | "lead" | "diff" => {
                let mut v = self.args(name, args, 1..=2, span)?;
                let n = match args.get(1) {
                    Some(a) => self.int_arg(name, a)?,
                    None => 1,
                };
                let x = v.remove(0);
                let order = self.order(name, span)?;
                let shifted = |e: Expr| -> Result<Expr, FormulaError> {
                    let f = if name == "lead" {
                        lead(e, Some(n), None)
                    } else {
                        lag(e, Some(n), None)
                    };
                    f.order_by(order.clone())
                        .build()
                        .map_err(|e| FormulaError::new(e.to_string(), span))
                };
                let e = if name == "diff" {
                    self.known(&x, args[0].span)?;
                    x.expr.clone() - shifted(x.expr.clone())?
                } else {
                    shifted(x.expr.clone())?
                };
                Ok(same_unit(x, e))
            }
            "cumsum" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                self.known(&x, args[0].span)?;
                let order = self.order(name, span)?;
                let frame = WindowFrame::new_bounds(
                    WindowFrameUnits::Rows,
                    WindowFrameBound::Preceding(ScalarValue::UInt64(None)),
                    WindowFrameBound::CurrentRow,
                );
                let running = window(sum_udaf(), x.expr.clone(), order, Some(frame), span)?;
                let e = when(x.expr.clone().is_null(), lit(ScalarValue::Null))
                    .otherwise(running)
                    .map_err(|e| FormulaError::new(e.to_string(), span))?;
                Ok(same_unit(x, e))
            }
            "rolling_mean" | "rolling_sum" | "rolling_min" | "rolling_max" => {
                let x = self.args(name, args, 2..=2, span)?.remove(0);
                self.known(&x, args[0].span)?;
                let n = self.int_arg(name, &args[1])?.max(1);
                let order = self.order(name, span)?;
                let frame = WindowFrame::new_bounds(
                    WindowFrameUnits::Rows,
                    WindowFrameBound::Preceding(ScalarValue::UInt64(Some(n as u64 - 1))),
                    WindowFrameBound::CurrentRow,
                );
                let udaf = match name {
                    "rolling_mean" => avg_udaf(),
                    "rolling_sum" => sum_udaf(),
                    "rolling_min" => min_udaf(),
                    _ => max_udaf(),
                };
                let arg = if name == "rolling_mean" {
                    f64(x.expr.clone())
                } else {
                    x.expr.clone()
                };
                let value = window(udaf, arg, order.clone(), Some(frame.clone()), span)?;
                let filled = window(count_udaf(), x.expr.clone(), order, Some(frame), span)?;
                let e = when(filled.gt_eq(lit(n)), value)
                    .end()
                    .map_err(|e| FormulaError::new(e.to_string(), span))?;
                Ok(same_unit(x, e))
            }
            "rank" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                let e = when(x.expr.clone().is_null(), lit(ScalarValue::Null))
                    .otherwise(
                        rank()
                            .order_by(vec![x.expr.clone().sort(true, false)])
                            .build()
                            .map_err(|e| FormulaError::new(e.to_string(), span))?,
                    )
                    .map_err(|e| FormulaError::new(e.to_string(), span))?;
                Ok(V::plain(e))
            }
            _ => err(format!("unknown function {name}()"), span),
        }
    }
}

/// An aggregate used as a window over the given order and frame (the whole column without them).
fn window(
    udaf: std::sync::Arc<datafusion::logical_expr::AggregateUDF>,
    arg: Expr,
    order: Vec<SortExpr>,
    frame: Option<WindowFrame>,
    span: Span,
) -> Result<Expr, FormulaError> {
    let f = Expr::from(WindowFunction::new(
        WindowFunctionDefinition::AggregateUDF(udaf),
        vec![arg],
    ));
    if order.is_empty() && frame.is_none() {
        return Ok(f);
    }
    let mut b = f.order_by(order);
    if let Some(frame) = frame {
        b = b.window_frame(frame);
    }
    b.build()
        .map_err(|e| FormulaError::new(e.to_string(), span))
}

/// The name closest to `name` by edit distance, if it's close enough to be a likely typo.
fn closest<'a>(name: &str, names: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let lower = name.to_lowercase();
    names
        .map(|n| (n, edit_distance(&lower, &n.to_lowercase())))
        .filter(|(n, d)| *d <= 2.max(n.len() / 4))
        .min_by_key(|(_, d)| *d)
        .map(|(n, _)| n)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != *cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::datatypes::{Field, Schema};
    use dv_model::field::Family;

    use super::*;

    fn dataset() -> Dataset {
        let with_unit = |name: &str, unit: &str| {
            Field::new(name, DataType::Float64, true)
                .with_metadata(HashMap::from([("units".to_string(), unit.to_string())]))
        };
        let schema = Schema::new(vec![
            with_unit("height_m", "m"),
            with_unit("height_ft", "ft"),
            with_unit("duration", "s"),
            with_unit("t2m", "K"),
            with_unit("weird", "furlongs"),
            Field::new("count", DataType::Int64, true),
        ]);
        Dataset::from_schema("d", Family::Table, &schema)
    }

    fn unit_of(src: &str) -> Result<Option<String>, FormulaError> {
        let ds = dataset();
        compile(src, &Context::new(&ds)).map(|f| f.unit.map(|u| u.base()))
    }

    #[test]
    fn units_propagate() {
        assert_eq!(
            unit_of("height_m + height_ft").unwrap().as_deref(),
            Some("m")
        );
        assert_eq!(
            unit_of("height_m / duration").unwrap().as_deref(),
            Some("m s-1")
        );
        assert_eq!(unit_of("height_m ^ 2").unwrap().as_deref(), Some("m2"));
        assert_eq!(
            unit_of("sqrt(height_m * height_ft)").unwrap().as_deref(),
            Some("m")
        );
        assert_eq!(
            unit_of("convert(t2m, \"degC\")").unwrap().as_deref(),
            Some("K")
        );
        assert_eq!(unit_of("count * 2").unwrap(), None);
        assert_eq!(unit_of("height_m > 3").unwrap(), None);
    }

    #[test]
    fn unit_errors() {
        let e = unit_of("height_m + duration").unwrap_err();
        assert!(e.message.contains("don't convert"), "{e}");
        let e = unit_of("1 + weird").unwrap_err();
        assert_eq!(e.span, (4, 9), "{e}");
        assert!(unit_of("exp(height_m)").is_err());
        assert!(unit_of("convert(count, \"m\")").is_err());
        assert!(unit_of("convert(height_m, \"s\")").is_err());
    }

    #[test]
    fn names_and_arguments() {
        let e = unit_of("heigth_m + 1").unwrap_err();
        assert!(e.message.contains("did you mean \"height_m\""), "{e}");
        assert!(unit_of("round(height_m, 1, 2)").is_err());
        assert!(unit_of("nope(1)").is_err());
        let e = unit_of("lag(count)").unwrap_err();
        assert!(e.message.contains("no time field"), "{e}");
    }
}
