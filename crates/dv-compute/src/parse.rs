//! The formula language's syntax: a small expression grammar, parsed with precedence climbing.
//!
//! ```text
//! tip / fare * 100
//! if(distance > 2, "long", "short")
//! [Trip distance] - mean([Trip distance])
//! convert(t2m, "degC")
//! ```
//!
//! Fields are bare names, or `[...]` or `` `...` `` when they contain spaces or symbols. Strings
//! use double or single quotes. Operators, loosest first: `or`/`||`, `and`/`&&`, `not`/`!`,
//! comparisons (`==` `!=` `<` `<=` `>` `>=`), `+` `-`, `*` `/` `%`, unary `-`, and `^` (right
//! associative). Every node records its byte span, so errors can point at the formula bar.

use std::fmt;

/// A byte range in the formula text.
pub type Span = (usize, usize);

/// A formula that can't be parsed or compiled, with where.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message} (at {}..{})", span.0, span.1)]
pub struct FormulaError {
    pub message: String,
    pub span: Span,
}

impl FormulaError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BinOp::*;
        f.write_str(match self {
            Or => "or",
            And => "and",
            Eq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Pow => "^",
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    Number(f64),
    Str(String),
    Bool(bool),
    Field(String),
    Neg(Box<Node>),
    Not(Box<Node>),
    Binary(BinOp, Box<Node>, Box<Node>),
    Call(String, Vec<Node>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub ast: Ast,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    /// A `[bracketed]` or `` `quoted` `` field name, which is never a keyword or function.
    Quoted(String),
    Op(&'static str),
    Open,
    Close,
    Comma,
}

fn lex(src: &str) -> Result<Vec<(Tok, Span)>, FormulaError> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let start = i;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let tok = if c.is_ascii_digit()
            || (c == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit))
        {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let text = &src[start..i];
            Tok::Num(
                text.parse()
                    .map_err(|_| FormulaError::new(format!("bad number {text}"), (start, i)))?,
            )
        } else if c == b'"' || c == b'\'' || c == b'[' || c == b'`' {
            let close = match c {
                b'[' => b']',
                q => q,
            };
            i += 1;
            let mut text = String::new();
            loop {
                match b.get(i) {
                    None => {
                        return Err(FormulaError::new("unclosed quote or bracket", (start, i)));
                    }
                    // A doubled closing quote is a literal quote.
                    Some(&q) if q == close && b.get(i + 1) == Some(&close) && close != b']' => {
                        text.push(q as char);
                        i += 2;
                    }
                    Some(&q) if q == close => {
                        i += 1;
                        break;
                    }
                    Some(_) => {
                        let ch = src[i..].chars().next().expect("in bounds");
                        text.push(ch);
                        i += ch.len_utf8();
                    }
                }
            }
            if c == b'"' || c == b'\'' {
                Tok::Str(text)
            } else {
                Tok::Quoted(text)
            }
        } else if c.is_ascii_alphabetic() || c == b'_' || c >= 0x80 {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] >= 0x80) {
                i += 1;
            }
            Tok::Ident(src[start..i].to_string())
        } else {
            let two = src.get(i..i + 2).unwrap_or("");
            let op = ["==", "!=", "<=", ">=", "&&", "||"]
                .into_iter()
                .find(|o| *o == two);
            if let Some(op) = op {
                i += 2;
                Tok::Op(op)
            } else {
                i += 1;
                match c {
                    b'(' => Tok::Open,
                    b')' => Tok::Close,
                    b',' => Tok::Comma,
                    b'+' => Tok::Op("+"),
                    b'-' => Tok::Op("-"),
                    b'*' => Tok::Op("*"),
                    b'/' => Tok::Op("/"),
                    b'%' => Tok::Op("%"),
                    b'^' => Tok::Op("^"),
                    b'<' => Tok::Op("<"),
                    b'>' => Tok::Op(">"),
                    b'!' => Tok::Op("!"),
                    b'=' => Tok::Op("=="),
                    _ => {
                        let ch = src[start..].chars().next().expect("in bounds");
                        return Err(FormulaError::new(
                            format!("unexpected {ch:?}"),
                            (start, start + ch.len_utf8()),
                        ));
                    }
                }
            }
        };
        out.push((tok, (start, i)));
    }
    Ok(out)
}

/// Parses a formula.
pub fn parse(src: &str) -> Result<Node, FormulaError> {
    let mut p = Parser {
        toks: lex(src)?,
        pos: 0,
        end: src.len(),
    };
    if p.toks.is_empty() {
        return Err(FormulaError::new("empty formula", (0, 0)));
    }
    let node = p.expr(0)?;
    if let Some((t, span)) = p.toks.get(p.pos) {
        return Err(FormulaError::new(format!("unexpected {t:?}"), *span));
    }
    Ok(node)
}

struct Parser {
    toks: Vec<(Tok, Span)>,
    pos: usize,
    end: usize,
}

/// Binding power of a binary operator: (left, right).
fn infix(tok: &Tok) -> Option<(BinOp, u8, u8)> {
    use BinOp::*;
    let (op, lbp) = match tok {
        Tok::Ident(w) if w.eq_ignore_ascii_case("or") => (Or, 1),
        Tok::Op("||") => (Or, 1),
        Tok::Ident(w) if w.eq_ignore_ascii_case("and") => (And, 3),
        Tok::Op("&&") => (And, 3),
        Tok::Op("==") => (Eq, 7),
        Tok::Op("!=") => (Ne, 7),
        Tok::Op("<") => (Lt, 7),
        Tok::Op("<=") => (Le, 7),
        Tok::Op(">") => (Gt, 7),
        Tok::Op(">=") => (Ge, 7),
        Tok::Op("+") => (Add, 9),
        Tok::Op("-") => (Sub, 9),
        Tok::Op("*") => (Mul, 11),
        Tok::Op("/") => (Div, 11),
        Tok::Op("%") => (Mod, 11),
        Tok::Op("^") => return Some((Pow, 15, 15)),
        _ => return None,
    };
    Some((op, lbp, lbp + 1))
}

const NOT_BP: u8 = 5;
const NEG_BP: u8 = 13;

impl Parser {
    fn next(&mut self) -> Option<(Tok, Span)> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(t, _)| t)
    }

    fn eof(&self) -> Span {
        (self.end, self.end)
    }

    fn expr(&mut self, min_bp: u8) -> Result<Node, FormulaError> {
        let mut lhs = self.prefix()?;
        while let Some(tok) = self.peek() {
            let Some((op, lbp, rbp)) = infix(tok) else {
                break;
            };
            if lbp < min_bp {
                break;
            }
            self.pos += 1;
            let rhs = self.expr(rbp)?;
            let span = (lhs.span.0, rhs.span.1);
            lhs = Node {
                ast: Ast::Binary(op, Box::new(lhs), Box::new(rhs)),
                span,
            };
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> Result<Node, FormulaError> {
        let Some((tok, span)) = self.next() else {
            return Err(FormulaError::new("formula ends early", self.eof()));
        };
        let node = |ast, end: Span| Node {
            ast,
            span: (span.0, end.1),
        };
        Ok(match tok {
            Tok::Num(n) => node(Ast::Number(n), span),
            Tok::Str(s) => node(Ast::Str(s), span),
            Tok::Quoted(s) => node(Ast::Field(s), span),
            Tok::Op("-") => {
                let inner = self.expr(NEG_BP)?;
                let end = inner.span;
                node(Ast::Neg(Box::new(inner)), end)
            }
            Tok::Op("!") => {
                let inner = self.expr(NOT_BP)?;
                let end = inner.span;
                node(Ast::Not(Box::new(inner)), end)
            }
            Tok::Ident(w) if w.eq_ignore_ascii_case("not") => {
                let inner = self.expr(NOT_BP)?;
                let end = inner.span;
                node(Ast::Not(Box::new(inner)), end)
            }
            Tok::Ident(w) if w.eq_ignore_ascii_case("true") => node(Ast::Bool(true), span),
            Tok::Ident(w) if w.eq_ignore_ascii_case("false") => node(Ast::Bool(false), span),
            Tok::Ident(name) if self.peek() == Some(&Tok::Open) => {
                self.pos += 1;
                let mut args = Vec::new();
                if self.peek() == Some(&Tok::Close) {
                    self.pos += 1;
                    return Ok(node(
                        Ast::Call(name.to_ascii_lowercase(), args),
                        self.toks[self.pos - 1].1,
                    ));
                }
                loop {
                    args.push(self.expr(0)?);
                    match self.next() {
                        Some((Tok::Comma, _)) => continue,
                        Some((Tok::Close, end)) => {
                            return Ok(node(Ast::Call(name.to_ascii_lowercase(), args), end));
                        }
                        Some((t, s)) => {
                            return Err(FormulaError::new(
                                format!("expected , or ) but found {t:?}"),
                                s,
                            ));
                        }
                        None => return Err(FormulaError::new("missing )", self.eof())),
                    }
                }
            }
            Tok::Ident(name) => node(Ast::Field(name), span),
            Tok::Open => {
                let inner = self.expr(0)?;
                match self.next() {
                    Some((Tok::Close, end)) => Node {
                        ast: inner.ast,
                        span: (span.0, end.1),
                    },
                    _ => return Err(FormulaError::new("missing )", self.eof())),
                }
            }
            t => return Err(FormulaError::new(format!("unexpected {t:?}"), span)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(n: &Node) -> String {
        match &n.ast {
            Ast::Number(x) => x.to_string(),
            Ast::Str(s) => format!("{s:?}"),
            Ast::Bool(b) => b.to_string(),
            Ast::Field(f) => format!("[{f}]"),
            Ast::Neg(a) => format!("(-{})", show(a)),
            Ast::Not(a) => format!("(not {})", show(a)),
            Ast::Binary(op, a, b) => format!("({} {op} {})", show(a), show(b)),
            Ast::Call(f, args) => {
                format!(
                    "{f}({})",
                    args.iter().map(show).collect::<Vec<_>>().join(", ")
                )
            }
        }
    }

    fn p(s: &str) -> String {
        show(&parse(s).unwrap())
    }

    #[test]
    fn precedence() {
        assert_eq!(p("a + b * c"), "([a] + ([b] * [c]))");
        assert_eq!(p("-a ^ 2"), "(-([a] ^ 2))");
        assert_eq!(p("2 ^ 3 ^ 2"), "(2 ^ (3 ^ 2))");
        assert_eq!(p("a - b - c"), "(([a] - [b]) - [c])");
        assert_eq!(
            p("not a > 1 and b or c"),
            "(((not ([a] > 1)) and [b]) or [c])"
        );
        assert_eq!(p("x == 1 || y != 'q'"), "(([x] == 1) or ([y] != \"q\"))");
    }

    #[test]
    fn names_strings_and_calls() {
        assert_eq!(
            p("[Trip distance] / `fare $`"),
            "([Trip distance] / [fare $])"
        );
        assert_eq!(
            p("IF(a > 1, \"say \"\"hi\"\"\", 'b')"),
            "if(([a] > 1), \"say \\\"hi\\\"\", \"b\")"
        );
        assert_eq!(p("now()"), "now()");
        assert_eq!(p("1.5e3 + .5"), "(1500 + 0.5)");
        assert_eq!(p("température + 1"), "([température] + 1)");
    }

    #[test]
    fn errors_point_at_the_problem() {
        assert_eq!(parse("a + ").unwrap_err().span, (4, 4));
        assert_eq!(parse("a $ b").unwrap_err().span, (2, 3));
        assert_eq!(parse("f(a b)").unwrap_err().span, (4, 5));
        assert!(parse("[open").is_err());
        assert!(parse("").is_err());
    }
}
