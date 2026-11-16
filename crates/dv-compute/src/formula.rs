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
//!
//! **Python.** Every formula also compiles to a polars expression with the same meaning, unit
//! conversions included, for exported scripts (see [`crate::export`]). Ordered functions use
//! `.over(order_by=...)`, so they don't depend on the frame's row order.

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

use crate::export::{py_float, py_str};
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
    /// The same formula as a polars expression (`pl` is polars), for exported scripts.
    pub python: String,
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
        python: v.py,
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
    /// The polars expression.
    py: String,
}

impl V {
    fn plain(expr: Expr, py: String) -> Self {
        V {
            expr,
            unit: U::None,
            py,
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

fn py_f64(py: &str) -> String {
    format!("{py}.cast(pl.Float64)")
}

/// The polars side of a unit conversion, with the same operations as the DataFusion side.
fn py_convert(py: &str, scale: f64, offset: f64) -> String {
    format!(
        "({} * {} + {})",
        py_f64(py),
        py_float(scale),
        py_float(offset)
    )
}

fn err<T>(msg: impl Into<String>, span: Span) -> Result<T, FormulaError> {
    Err(FormulaError::new(msg, span))
}

impl Compiler<'_, '_> {
    fn node(&mut self, n: &Node) -> Result<V, FormulaError> {
        match &n.ast {
            Ast::Number(x) => Ok(V::plain(lit(*x), format!("pl.lit({})", py_float(*x)))),
            Ast::Str(s) => Ok(V::plain(lit(s.clone()), format!("pl.lit({})", py_str(s)))),
            Ast::Bool(b) => Ok(V::plain(
                lit(*b),
                format!("pl.lit({})", if *b { "True" } else { "False" }),
            )),
            Ast::Field(name) => self.field(name, n.span),
            Ast::Neg(a) => {
                let v = self.node(a)?;
                Ok(V {
                    expr: -v.expr,
                    unit: v.unit,
                    py: format!("(-{})", v.py),
                })
            }
            Ast::Not(a) => {
                let v = self.node(a)?;
                Ok(V::plain(!v.expr, format!("(~{})", v.py)))
            }
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
            py: format!("pl.col({})", py_str(name)),
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
    fn align(&self, a: &V, b: V, op: &str, span: Span) -> Result<(U, Expr, String), FormulaError> {
        Ok(match (&a.unit, b.unit) {
            (U::Known(ua), U::Known(ub)) => match ub.converter(ua) {
                Some(c) if c.is_identity() => (a.unit.clone(), b.expr, b.py),
                Some(c) => (
                    a.unit.clone(),
                    f64(b.expr) * lit(c.scale) + lit(c.offset),
                    py_convert(&b.py, c.scale, c.offset),
                ),
                None => {
                    return err(
                        format!("can't {op} {ub} and {ua}: the units don't convert"),
                        span,
                    );
                }
            },
            (U::Known(_), U::None) => (a.unit.clone(), b.expr, b.py),
            (U::None, u @ U::Known(_)) => (u, b.expr, b.py),
            _ => (U::None, b.expr, b.py),
        })
    }

    fn binary(&mut self, op: BinOp, a: &Node, b: &Node, span: Span) -> Result<V, FormulaError> {
        let va = self.node(a)?;
        let vb = self.node(b)?;
        use BinOp::*;
        if matches!(op, And | Or) {
            let (e, py) = if op == And {
                (va.expr.and(vb.expr), format!("({} & {})", va.py, vb.py))
            } else {
                (va.expr.or(vb.expr), format!("({} | {})", va.py, vb.py))
            };
            return Ok(V::plain(e, py));
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
                let left_py = va.py.clone();
                let (unit, right, right_py) = self.align(&va, vb, verb, span)?;
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
                V {
                    expr: e,
                    unit,
                    py: format!("({left_py} {op} {right_py})"),
                }
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
                let (e, py) = if op == Mul {
                    (va.expr * vb.expr, format!("({} * {})", va.py, vb.py))
                } else {
                    (
                        f64(va.expr) / f64(vb.expr),
                        format!("({} / {})", py_f64(&va.py), py_f64(&vb.py)),
                    )
                };
                V { expr: e, unit, py }
            }
            // DataFusion's remainder takes the dividend's sign, as in Rust and C; polars' takes
            // the divisor's, as in Python. The difference is one divisor.
            Mod => {
                let (a, b) = (&va.py, &vb.py);
                let r = format!("({a} % {b})");
                V {
                    expr: va.expr % vb.expr,
                    unit: va.unit,
                    py: format!(
                        "pl.when(({r} != 0) & (({r} < 0) != ({a} < 0))).then({r} - {b}).otherwise({r})"
                    ),
                }
            }
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
                    py: format!("({} ** {})", py_f64(&va.py), py_f64(&vb.py)),
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

    fn order(&self, name: &str, span: Span) -> Result<(Vec<SortExpr>, String), FormulaError> {
        match self.ctx.order_by {
            // Nulls first on both sides: polars' `over` puts them first by default.
            Some(f) => Ok((
                vec![ident(f).sort(true, true)],
                format!(".over(None, order_by={})", py_str(f)),
            )),
            None => err(
                format!("{name}() needs an order; this dataset has no time field to order by"),
                span,
            ),
        }
    }

    fn call(&mut self, name: &str, args: &[Node], span: Span) -> Result<V, FormulaError> {
        let same_unit = |v: V, e: Expr, py: String| V {
            expr: e,
            unit: v.unit,
            py,
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
                let (e, py) = match name {
                    "abs" => (math::abs(x.expr.clone()), format!("{}.abs()", x.py)),
                    "floor" => (
                        math::floor(f64(x.expr.clone())),
                        format!("{}.floor()", py_f64(&x.py)),
                    ),
                    "ceil" => (
                        math::ceil(f64(x.expr.clone())),
                        format!("{}.ceil()", py_f64(&x.py)),
                    ),
                    _ => {
                        let digits = match args.get(1) {
                            Some(a) => self.int_arg(name, a)?,
                            None => 0,
                        };
                        // DataFusion rounds halves away from zero; polars rounds them to even
                        // unless told otherwise.
                        (
                            math::round(vec![f64(x.expr.clone()), lit(digits)]),
                            format!(
                                "{}.round({digits}, mode=\"half_away_from_zero\")",
                                py_f64(&x.py)
                            ),
                        )
                    }
                };
                Ok(same_unit(x, e, py))
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
                    py: format!("{}.sqrt()", py_f64(&x.py)),
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
                let py = py_f64(&x.py);
                Ok(match name {
                    "exp" => V::plain(math::exp(e), format!("{py}.exp()")),
                    "ln" => V::plain(math::ln(e), format!("{py}.log()")),
                    _ => V::plain(math::log10(e), format!("{py}.log10()")),
                })
            }
            "min" | "max" if args.len() == 2 => {
                let mut v = self.args(name, args, 2..=2, span)?;
                let b = v.pop().expect("two");
                let a = v.pop().expect("two");
                self.known(&a, args[0].span)?;
                self.known(&b, args[1].span)?;
                let left = a.expr.clone();
                let left_py = a.py.clone();
                let (unit, right, right_py) = self.align(&a, b, "compare", span)?;
                let e = if name == "min" {
                    least(vec![left, right])
                } else {
                    greatest(vec![left, right])
                };
                Ok(V {
                    expr: e,
                    unit,
                    py: format!("pl.{name}_horizontal({left_py}, {right_py})"),
                })
            }
            "coalesce" => {
                let v = self.args(name, args, 1..=usize::MAX, span)?;
                let mut it = v.into_iter();
                let first = it.next().expect("at least one");
                let mut exprs = vec![first.expr.clone()];
                let mut pys = vec![first.py.clone()];
                let mut unit = first.unit.clone();
                for (v, a) in it.zip(&args[1..]) {
                    self.known(&v, a.span)?;
                    let base = V {
                        expr: lit(0),
                        unit: unit.clone(),
                        py: String::new(),
                    };
                    let (u, e, py) = self.align(&base, v, "combine", a.span)?;
                    unit = u;
                    exprs.push(e);
                    pys.push(py);
                }
                Ok(V {
                    expr: coalesce(exprs),
                    unit,
                    py: format!("pl.coalesce({})", pys.join(", ")),
                })
            }
            "if" => {
                let mut v = self.args(name, args, 3..=3, span)?;
                let otherwise = v.pop().expect("three");
                let then = v.pop().expect("three");
                let cond = v.pop().expect("three");
                let then_expr = then.expr.clone();
                let then_py = then.py.clone();
                let (unit, else_expr, else_py) =
                    self.align(&then, otherwise, "combine", args[2].span)?;
                let e = when(cond.expr, then_expr)
                    .otherwise(else_expr)
                    .map_err(|e| FormulaError::new(e.to_string(), span))?;
                Ok(V {
                    expr: e,
                    unit,
                    py: format!("pl.when({}).then({then_py}).otherwise({else_py})", cond.py),
                })
            }
            "is_null" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                Ok(V::plain(x.expr.is_null(), format!("{}.is_null()", x.py)))
            }
            "year" | "month" | "day" | "hour" | "minute" | "weekday" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                let part = if name == "weekday" { "dow" } else { name };
                // DataFusion's `dow` counts from Sunday = 0; polars' weekday from Monday = 1.
                let py = if name == "weekday" {
                    format!("({}.dt.weekday() % 7)", x.py)
                } else {
                    format!("{}.dt.{name}()", x.py)
                };
                Ok(V::plain(
                    cast(date_part(lit(part), x.expr), DataType::Int64),
                    format!("{py}.cast(pl.Int64)"),
                ))
            }
            "lower" | "upper" | "length" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                Ok(match name {
                    "lower" => V::plain(lower(x.expr), format!("{}.str.to_lowercase()", x.py)),
                    "upper" => V::plain(upper(x.expr), format!("{}.str.to_uppercase()", x.py)),
                    _ => V::plain(
                        character_length(x.expr),
                        format!("{}.str.len_chars()", x.py),
                    ),
                })
            }
            "contains" => {
                let mut v = self.args(name, args, 2..=2, span)?;
                let needle = v.pop().expect("two");
                let hay = v.pop().expect("two");
                Ok(V::plain(
                    contains(hay.expr, needle.expr),
                    format!("{}.str.contains({}, literal=True)", hay.py, needle.py),
                ))
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
                    py: py_convert(&x.py, c.scale, c.offset),
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
                let (arg, arg_py) = if matches!(name, "mean" | "std" | "var") {
                    (f64(x.expr), py_f64(&x.py))
                } else {
                    (x.expr, x.py)
                };
                let e = window(udaf, arg, vec![], None, span)?;
                let unit = match (name, x.unit) {
                    ("count", _) => U::None,
                    ("var", U::Known(u)) => U::Known(u.powi(2)),
                    (_, u) => u,
                };
                // Sample standard deviation and variance, as DataFusion's `stddev` and
                // `var_samp`: polars' default `ddof=1`.
                Ok(V {
                    expr: e,
                    unit,
                    py: format!("{arg_py}.{name}()"),
                })
            }
            "lag" | "lead" | "diff" => {
                let mut v = self.args(name, args, 1..=2, span)?;
                let n = match args.get(1) {
                    Some(a) => self.int_arg(name, a)?,
                    None => 1,
                };
                let x = v.remove(0);
                let (order, over) = self.order(name, span)?;
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
                let shift = if name == "lead" { -n } else { n };
                let shifted_py = format!("{}.shift({shift}){over}", x.py);
                let (e, py) = if name == "diff" {
                    self.known(&x, args[0].span)?;
                    (
                        x.expr.clone() - shifted(x.expr.clone())?,
                        format!("({} - {shifted_py})", x.py),
                    )
                } else {
                    (shifted(x.expr.clone())?, shifted_py)
                };
                Ok(same_unit(x, e, py))
            }
            "cumsum" => {
                let x = self.args(name, args, 1..=1, span)?.remove(0);
                self.known(&x, args[0].span)?;
                let (order, over) = self.order(name, span)?;
                let frame = WindowFrame::new_bounds(
                    WindowFrameUnits::Rows,
                    WindowFrameBound::Preceding(ScalarValue::UInt64(None)),
                    WindowFrameBound::CurrentRow,
                );
                let running = window(sum_udaf(), x.expr.clone(), order, Some(frame), span)?;
                let e = when(x.expr.clone().is_null(), lit(ScalarValue::Null))
                    .otherwise(running)
                    .map_err(|e| FormulaError::new(e.to_string(), span))?;
                // polars' cum_sum already leaves nulls where the input is null.
                let py = format!("{}.cum_sum(){over}", x.py);
                Ok(same_unit(x, e, py))
            }
            "rolling_mean" | "rolling_sum" | "rolling_min" | "rolling_max" => {
                let x = self.args(name, args, 2..=2, span)?.remove(0);
                self.known(&x, args[0].span)?;
                let n = self.int_arg(name, &args[1])?.max(1);
                let (order, over) = self.order(name, span)?;
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
                let (arg, arg_py) = if name == "rolling_mean" {
                    (f64(x.expr.clone()), py_f64(&x.py))
                } else {
                    (x.expr.clone(), x.py.clone())
                };
                let value = window(udaf, arg, order.clone(), Some(frame.clone()), span)?;
                let filled = window(count_udaf(), x.expr.clone(), order, Some(frame), span)?;
                let e = when(filled.gt_eq(lit(n)), value)
                    .end()
                    .map_err(|e| FormulaError::new(e.to_string(), span))?;
                // polars' rolling windows need `window_size` non-null values by default, as here.
                let py = format!("{arg_py}.{name}({n}){over}");
                Ok(same_unit(x, e, py))
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
                // polars ranks only non-null values and leaves nulls null.
                Ok(V::plain(e, format!("{}.rank(\"min\")", x.py)))
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
