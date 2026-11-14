// Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
//! Column alignment and table assembly: `_align_column`, `_align_header`,
//! `_align_cell_veritically`, `_format_table` and the built-in `TableFormat`s, line by line.
//!
//! Widths are `isize`: `wcwidth.wcswidth` returns -1 for control characters, and tabulate
//! carries that through (`"-" * -1` is `""`). The [`Host`] measures text that is not
//! printable ASCII, strips ANSI codes the native matcher leaves to Python, and runs custom
//! `TableFormat` callables.

use std::borrow::Cow;

use super::ansi;
use super::number::afterpoint;
use super::pystr::{
    self, is_printable_ascii, push_spaces, py_len, rstrip, split_cr_lf, splitlines,
};

/// A column (or header) alignment as tabulate compares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Right,
    Center,
    Decimal,
    Left,
    /// Any other truthy value (`"weird"`, `True`): padded like `"left"` but not equal to it.
    Other,
    /// A falsy value (`None`, `""`, `False`): no padding.
    Falsy,
}

/// `rowalign` values: `"bottom"`, `"center"`, anything else is the top.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowAlign {
    Top,
    Center,
    Bottom,
}

/// Which alignment list a custom callable receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignList {
    Columns,
    Headers,
}

/// What a line or row producer gave: a `str`, or another object (its type name), which
/// makes `"\n".join(lines)` raise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    Str(String),
    Other(String),
}

/// A row handed to a row producer: cells, or tabulate's `SEPARATING_LINE` string itself.
#[derive(Clone, Copy, Debug)]
pub enum Cells<'a> {
    List(&'a [String]),
    Separator,
}

/// What the table needs from the interpreter.
pub trait Host {
    type Error;
    /// `wcwidth.wcswidth(s)` (only called for text that is not printable ASCII).
    fn wcswidth(&mut self, s: &str) -> Result<isize, Self::Error>;
    /// `_strip_ansi(s)` where the native matcher cannot decide.
    fn strip_ansi(&mut self, s: &str) -> Result<String, Self::Error>;
    /// `_afterpoint(s)` where the native test cannot decide.
    fn afterpoint(&mut self, s: &str) -> Result<isize, Self::Error>;
    /// A custom line callable: `linefmt(colwidths, colaligns)`.
    fn line(
        &mut self,
        id: usize,
        colwidths: &[isize],
        aligns: AlignList,
    ) -> Result<Piece, Self::Error>;
    /// A custom row callable: `rowfmt(cells, colwidths, colaligns)`.
    fn row(
        &mut self,
        id: usize,
        cells: Cells<'_>,
        colwidths: &[isize],
        aligns: AlignList,
    ) -> Result<Piece, Self::Error>;
}

/// Errors tabulate raises from these steps.
#[derive(Debug)]
pub enum Error<E> {
    Host(E),
    /// `"{0:>%ds}" % width` with a negative width: `ValueError`.
    NegativeWidth,
    /// `_asciidoc_row`'s `alignment[colalign]`: `KeyError` for the alignment at this index.
    AsciidocKey {
        list: AlignList,
        index: usize,
    },
    /// `_textile_row_with_attrs` on `SEPARATING_LINE`: `TypeError`.
    TextileSeparator,
    /// `_textile_row_with_attrs` on an empty row: `IndexError`.
    TextileEmpty,
    /// `rowaligns[-1]` on an empty list: `IndexError`.
    RowAlignIndex,
    /// `max()` of an empty sequence: `ValueError`.
    EmptyMax,
    /// `"\n".join(lines)` met a non-str: `TypeError`.
    Join {
        index: usize,
        type_name: String,
    },
}

impl<E> From<E> for Error<E> {
    fn from(e: E) -> Self {
        Self::Host(e)
    }
}

/// A horizontal line of a `TableFormat`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LineFmt {
    /// `None` or another falsy value.
    Off,
    /// `Line(begin, hline, sep, end)`.
    Plain {
        begin: String,
        fill: String,
        sep: String,
        end: String,
    },
    /// `_pipe_line_with_colons`.
    PipeColons,
    /// `_grid_line_with_colons`.
    GridColons,
    /// `_html_begin_table_without_header`.
    HtmlBegin,
    /// `_latex_line_begin_tabular`.
    LatexBegin { booktabs: bool, longtable: bool },
    /// `partial(_asciidoc_row, False)` as a line.
    Asciidoc,
    /// Any other callable, run by the host.
    Custom(usize),
}

/// A row of a `TableFormat`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowFmt {
    Off,
    /// `DataRow(begin, sep, end)`.
    Plain {
        begin: String,
        sep: String,
        end: String,
    },
    /// `partial(_mediawiki_row_with_attrs, sep)`.
    Mediawiki(char),
    /// `partial(_moin_row_with_attrs, "||"[, header="'''"])`.
    Moin {
        header: bool,
    },
    /// `partial(_html_row_with_attrs, "th"|"td", unsafe)`.
    Html {
        th: bool,
        unsafe_: bool,
    },
    /// `_latex_row` (`raw`: `escrules={}`).
    Latex {
        raw: bool,
    },
    /// `_textile_row_with_attrs`.
    Textile,
    /// `partial(_asciidoc_row, is_header)`.
    Asciidoc {
        header: bool,
    },
    Custom(usize),
}

impl LineFmt {
    fn on(&self) -> bool {
        *self != Self::Off
    }

    fn plain(begin: &str, fill: &str, sep: &str, end: &str) -> Self {
        Self::Plain {
            begin: begin.into(),
            fill: fill.into(),
            sep: sep.into(),
            end: end.into(),
        }
    }
}

impl RowFmt {
    fn plain(begin: &str, sep: &str, end: &str) -> Self {
        Self::Plain {
            begin: begin.into(),
            sep: sep.into(),
            end: end.into(),
        }
    }
}

/// A `TableFormat`, with `with_header_hide` resolved to four flags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableFormat {
    pub lineabove: LineFmt,
    pub linebelowheader: LineFmt,
    pub linebetweenrows: LineFmt,
    pub linebelow: LineFmt,
    pub headerrow: RowFmt,
    pub datarow: RowFmt,
    pub padding: isize,
    /// `"lineabove" in with_header_hide`, and so on (applied only when there are headers).
    pub hide: [bool; 4],
}

const ABOVE: usize = 0;
const BELOW_HEADER: usize = 1;
const BETWEEN: usize = 2;
const BELOW: usize = 3;

/// The names of tabulate 0.10.0's built-in formats (`tabulate_formats`, sorted).
pub const FORMAT_NAMES: [&str; 37] = [
    "asciidoc",
    "colon_grid",
    "double_grid",
    "double_outline",
    "fancy_grid",
    "fancy_outline",
    "github",
    "grid",
    "heavy_grid",
    "heavy_outline",
    "html",
    "jira",
    "latex",
    "latex_booktabs",
    "latex_longtable",
    "latex_raw",
    "mediawiki",
    "mixed_grid",
    "mixed_outline",
    "moinmoin",
    "orgtbl",
    "outline",
    "pipe",
    "plain",
    "presto",
    "pretty",
    "psql",
    "rounded_grid",
    "rounded_outline",
    "rst",
    "simple",
    "simple_grid",
    "simple_outline",
    "textile",
    "tsv",
    "unsafehtml",
    "youtrack",
];

/// A built-in format by name.
#[allow(clippy::too_many_lines)]
pub fn builtin(name: &str) -> Option<TableFormat> {
    use LineFmt as L;
    use RowFmt as R;
    let grid = |a: [&str; 4], h: [&str; 4], b: [&str; 4], z: [&str; 4], v: &str, outline: bool| {
        TableFormat {
            lineabove: L::plain(a[0], a[1], a[2], a[3]),
            linebelowheader: L::plain(h[0], h[1], h[2], h[3]),
            linebetweenrows: if outline {
                L::Off
            } else {
                L::plain(b[0], b[1], b[2], b[3])
            },
            linebelow: L::plain(z[0], z[1], z[2], z[3]),
            headerrow: R::plain(v, v, v),
            datarow: R::plain(v, v, v),
            padding: 1,
            hide: [false; 4],
        }
    };
    let simple_dash = || L::plain("", "-", "  ", "");
    let f = match name {
        "simple" => TableFormat {
            lineabove: simple_dash(),
            linebelowheader: simple_dash(),
            linebetweenrows: L::Off,
            linebelow: simple_dash(),
            headerrow: R::plain("", "  ", ""),
            datarow: R::plain("", "  ", ""),
            padding: 0,
            hide: [true, false, false, true],
        },
        "plain" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::Off,
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("", "  ", ""),
            datarow: R::plain("", "  ", ""),
            padding: 0,
            hide: [false; 4],
        },
        "grid" | "outline" => {
            let d = ["+", "-", "+", "+"];
            grid(d, ["+", "=", "+", "+"], d, d, "|", name == "outline")
        }
        "simple_grid" | "simple_outline" => grid(
            ["┌", "─", "┬", "┐"],
            ["├", "─", "┼", "┤"],
            ["├", "─", "┼", "┤"],
            ["└", "─", "┴", "┘"],
            "│",
            name == "simple_outline",
        ),
        "rounded_grid" | "rounded_outline" => grid(
            ["╭", "─", "┬", "╮"],
            ["├", "─", "┼", "┤"],
            ["├", "─", "┼", "┤"],
            ["╰", "─", "┴", "╯"],
            "│",
            name == "rounded_outline",
        ),
        "heavy_grid" | "heavy_outline" => grid(
            ["┏", "━", "┳", "┓"],
            ["┣", "━", "╋", "┫"],
            ["┣", "━", "╋", "┫"],
            ["┗", "━", "┻", "┛"],
            "┃",
            name == "heavy_outline",
        ),
        "mixed_grid" | "mixed_outline" => grid(
            ["┍", "━", "┯", "┑"],
            ["┝", "━", "┿", "┥"],
            ["├", "─", "┼", "┤"],
            ["┕", "━", "┷", "┙"],
            "│",
            name == "mixed_outline",
        ),
        "double_grid" | "double_outline" => grid(
            ["╔", "═", "╦", "╗"],
            ["╠", "═", "╬", "╣"],
            ["╠", "═", "╬", "╣"],
            ["╚", "═", "╩", "╝"],
            "║",
            name == "double_outline",
        ),
        "fancy_grid" | "fancy_outline" => grid(
            ["╒", "═", "╤", "╕"],
            ["╞", "═", "╪", "╡"],
            ["├", "─", "┼", "┤"],
            ["╘", "═", "╧", "╛"],
            "│",
            name == "fancy_outline",
        ),
        "colon_grid" => {
            let d = ["+", "-", "+", "+"];
            let mut g = grid(d, d, d, d, "|", false);
            g.linebelowheader = L::GridColons;
            g
        }
        "github" => TableFormat {
            lineabove: L::plain("|", "-", "|", "|"),
            linebelowheader: L::plain("|", "-", "|", "|"),
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("|", "|", "|"),
            datarow: R::plain("|", "|", "|"),
            padding: 1,
            hide: [true, false, false, false],
        },
        "pipe" => TableFormat {
            lineabove: L::PipeColons,
            linebelowheader: L::PipeColons,
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("|", "|", "|"),
            datarow: R::plain("|", "|", "|"),
            padding: 1,
            hide: [true, false, false, false],
        },
        "orgtbl" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::plain("|", "-", "+", "|"),
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("|", "|", "|"),
            datarow: R::plain("|", "|", "|"),
            padding: 1,
            hide: [false; 4],
        },
        "jira" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::Off,
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("||", "||", "||"),
            datarow: R::plain("|", "|", "|"),
            padding: 1,
            hide: [false; 4],
        },
        "presto" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::plain("", "-", "+", ""),
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("", "|", ""),
            datarow: R::plain("", "|", ""),
            padding: 1,
            hide: [false; 4],
        },
        "pretty" => {
            let d = ["+", "-", "+", "+"];
            grid(d, d, d, d, "|", true)
        }
        "psql" => {
            let d = ["+", "-", "+", "+"];
            grid(d, ["|", "-", "+", "|"], d, d, "|", true)
        }
        "rst" => TableFormat {
            lineabove: L::plain("", "=", "  ", ""),
            linebelowheader: L::plain("", "=", "  ", ""),
            linebetweenrows: L::Off,
            linebelow: L::plain("", "=", "  ", ""),
            headerrow: R::plain("", "  ", ""),
            datarow: R::plain("", "  ", ""),
            padding: 0,
            hide: [false; 4],
        },
        "mediawiki" => TableFormat {
            lineabove: L::plain(
                "{| class=\"wikitable\" style=\"text-align: left;\"",
                "",
                "",
                "\n|+ <!-- caption -->\n|-",
            ),
            linebelowheader: L::plain("|-", "", "", ""),
            linebetweenrows: L::plain("|-", "", "", ""),
            linebelow: L::plain("|}", "", "", ""),
            headerrow: R::Mediawiki('!'),
            datarow: R::Mediawiki('|'),
            padding: 0,
            hide: [false; 4],
        },
        "moinmoin" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::Off,
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::Moin { header: true },
            datarow: R::Moin { header: false },
            padding: 1,
            hide: [false; 4],
        },
        "youtrack" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::Off,
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("|| ", " || ", " || "),
            datarow: R::plain("| ", " | ", " |"),
            padding: 1,
            hide: [false; 4],
        },
        "html" | "unsafehtml" => {
            let unsafe_ = name == "unsafehtml";
            TableFormat {
                lineabove: L::HtmlBegin,
                linebelowheader: L::Off,
                linebetweenrows: L::Off,
                linebelow: L::plain("</tbody>\n</table>", "", "", ""),
                headerrow: R::Html { th: true, unsafe_ },
                datarow: R::Html { th: false, unsafe_ },
                padding: 0,
                hide: [true, false, false, false],
            }
        }
        "latex" | "latex_raw" | "latex_booktabs" | "latex_longtable" => {
            let booktabs = name == "latex_booktabs";
            let longtable = name == "latex_longtable";
            let raw = name == "latex_raw";
            let (below_header, below) = if booktabs {
                ("\\midrule", "\\bottomrule\n\\end{tabular}")
            } else if longtable {
                ("\\hline\n\\endhead", "\\hline\n\\end{longtable}")
            } else {
                ("\\hline", "\\hline\n\\end{tabular}")
            };
            TableFormat {
                lineabove: L::LatexBegin {
                    booktabs,
                    longtable,
                },
                linebelowheader: L::plain(below_header, "", "", ""),
                linebetweenrows: L::Off,
                linebelow: L::plain(below, "", "", ""),
                headerrow: R::Latex { raw },
                datarow: R::Latex { raw },
                padding: 1,
                hide: [false; 4],
            }
        }
        "tsv" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::Off,
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("", "\t", ""),
            datarow: R::plain("", "\t", ""),
            padding: 0,
            hide: [false; 4],
        },
        "textile" => TableFormat {
            lineabove: L::Off,
            linebelowheader: L::Off,
            linebetweenrows: L::Off,
            linebelow: L::Off,
            headerrow: R::plain("|_. ", "|_.", "|"),
            datarow: R::Textile,
            padding: 1,
            hide: [false; 4],
        },
        "asciidoc" => TableFormat {
            lineabove: L::Asciidoc,
            linebelowheader: L::Off,
            linebetweenrows: L::Off,
            linebelow: L::plain("|====", "", "", ""),
            headerrow: R::Asciidoc { header: true },
            datarow: R::Asciidoc { header: false },
            padding: 1,
            hide: [true, false, false, false],
        },
        _ => return None,
    };
    Some(f)
}

/// `simple_separated_format(sep)`.
pub fn simple_separated(sep: &str) -> TableFormat {
    TableFormat {
        lineabove: LineFmt::Off,
        linebelowheader: LineFmt::Off,
        linebetweenrows: LineFmt::Off,
        linebelow: LineFmt::Off,
        headerrow: RowFmt::plain("", sep, ""),
        datarow: RowFmt::plain("", sep, ""),
        padding: 0,
        hide: [false; 4],
    }
}

/// The per-table settings the measuring and aligning steps share (tabulate's flags).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub has_invisible: bool,
    pub wide: bool,
    pub is_multiline: bool,
    pub preserve_whitespace: bool,
    pub max_digits: usize,
    pub unicode_ok: bool,
}

/// The table engine: tabulate's width functions, aligners and `_format_table`.
#[derive(Debug)]
pub struct Engine<'h, H> {
    pub host: &'h mut H,
    pub s: Settings,
}

type Res<T, E> = Result<T, Error<E>>;

/// Padding functions (`_padleft`, `_padright`, `_padboth`, `_padnone`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pad {
    Left,
    Right,
    Both,
    None,
}

/// A cell after `_align_column`: the string `" " * left + text + " " * right`, kept apart so
/// the padding is written straight into the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Aligned<'a> {
    pub text: Cow<'a, str>,
    pub left: usize,
    pub right: usize,
}

impl Aligned<'_> {
    /// The padded string.
    pub fn padded(&self) -> Cow<'_, str> {
        if self.left == 0 && self.right == 0 {
            return Cow::Borrowed(&self.text);
        }
        let mut s = String::with_capacity(self.text.len() + self.left + self.right);
        s.extend(std::iter::repeat_n(' ', self.left));
        s.push_str(&self.text);
        s.extend(std::iter::repeat_n(' ', self.right));
        Cow::Owned(s)
    }

    fn write(&self, buf: &mut String, extra: isize) {
        let extra = usize::try_from(extra).unwrap_or(0);
        buf.extend(std::iter::repeat_n(' ', self.left + extra));
        buf.push_str(&self.text);
        buf.extend(std::iter::repeat_n(' ', self.right + extra));
    }
}

/// `"{0:>%ds}" % width` and friends.
fn pad<E>(kind: Pad, width: isize, s: Cow<'_, str>) -> Res<Aligned<'_>, E> {
    let none = |text| {
        Ok(Aligned {
            text,
            left: 0,
            right: 0,
        })
    };
    if kind == Pad::None {
        return none(s);
    }
    if width < 0 {
        return Err(Error::NegativeWidth);
    }
    let n = width - len_i(&s);
    let Ok(n) = usize::try_from(n) else {
        return none(s);
    };
    let (left, right) = match kind {
        Pad::Left => (n, 0),
        Pad::Right => (0, n),
        Pad::Both => (n / 2, n - n / 2),
        Pad::None => unreachable!(),
    };
    Ok(Aligned {
        text: s,
        left,
        right,
    })
}

fn len_i(s: &str) -> isize {
    isize::try_from(py_len(s)).unwrap_or(isize::MAX)
}

/// `s.strip()` without copying unless something is removed from an owned string.
fn strip_cow(s: Cow<'_, str>) -> Cow<'_, str> {
    match s {
        Cow::Borrowed(b) => Cow::Borrowed(pystr::strip(b)),
        Cow::Owned(o) => {
            let t = pystr::strip(&o);
            if t.len() == o.len() {
                Cow::Owned(o)
            } else {
                Cow::Owned(t.to_owned())
            }
        }
    }
}

/// A data row of the finished table, by position, or a re-inserted `SEPARATING_LINE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRef {
    Data(usize),
    Separator,
}

/// The cells of one row as `_format_table` passes them on.
#[derive(Clone, Copy)]
enum Src<'r> {
    Cols(&'r [Vec<Aligned<'r>>], usize),
    List(&'r [String]),
    Separator,
}

impl Src<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Cols(c, _) => c.len(),
            Self::List(l) => l.len(),
            Self::Separator => 1,
        }
    }

    /// Cell `j` as tabulate holds it (with its alignment padding).
    fn get(&self, j: usize) -> Cow<'_, str> {
        match self {
            Self::Cols(c, i) => c[j][*i].padded(),
            Self::List(l) => Cow::Borrowed(&l[j]),
            Self::Separator => Cow::Borrowed("\u{1}"),
        }
    }

    /// Append cell `j` with `padding` more spaces on each side.
    fn write(&self, j: usize, buf: &mut String, padding: isize) {
        if let Self::Cols(c, i) = self {
            c[j][*i].write(buf, padding);
        } else {
            push_spaces(buf, padding);
            buf.push_str(&self.get(j));
            push_spaces(buf, padding);
        }
    }

    /// Cell `j` after `_pad_row(cells, padding)`, as (spaces, text, spaces).
    #[allow(clippy::many_single_char_names)]
    fn parts(&self, j: usize, padding: isize) -> (usize, &str, usize) {
        let p = usize::try_from(padding).unwrap_or(0);
        match self {
            Self::Cols(c, i) => {
                let a = &c[j][*i];
                (a.left + p, &a.text, a.right + p)
            }
            Self::List(l) => (p, &l[j], p),
            Self::Separator => (0, "\u{1}", 0),
        }
    }

    /// `_pad_row(cells, padding)` as owned strings (the separator stays itself).
    fn padded(&self, padding: isize) -> Vec<String> {
        (0..self.len())
            .map(|j| {
                let mut s = String::new();
                self.write(j, &mut s, padding);
                s
            })
            .collect()
    }
}

/// `lines` and `"\n".join(lines)` in one buffer; the first non-str line is remembered.
struct Out {
    buf: String,
    count: usize,
    bad: Option<(usize, String)>,
}

impl Out {
    fn start(&mut self) -> usize {
        if self.count > 0 {
            self.buf.push('\n');
        }
        self.count += 1;
        self.buf.len()
    }

    fn rstrip_from(&mut self, start: usize) {
        let keep = start + rstrip(&self.buf[start..]).len();
        self.buf.truncate(keep);
    }

    fn piece(&mut self, p: Piece) {
        match p {
            Piece::Str(s) => {
                self.start();
                self.buf.push_str(&s);
            }
            Piece::Other(type_name) => {
                if self.bad.is_none() {
                    self.bad = Some((self.count, type_name));
                }
                self.count += 1;
            }
        }
    }
}

impl<H: Host> Engine<'_, H> {
    /// `_strip_ansi(s)`.
    pub fn strip_ansi<'s>(&mut self, s: &'s str) -> Res<Cow<'s, str>, H::Error> {
        match ansi::strip(s) {
            Ok(None) => Ok(Cow::Borrowed(s)),
            Ok(Some(t)) => Ok(Cow::Owned(t)),
            Err(()) => Ok(Cow::Owned(self.host.strip_ansi(s)?)),
        }
    }

    /// The width of one line of text: `_visible_width`, `wcwidth.wcswidth` or `len`.
    pub fn line_width(&mut self, s: &str) -> Res<isize, H::Error> {
        if is_printable_ascii(s) {
            return Ok(len_i(s));
        }
        let t = if self.s.has_invisible {
            self.strip_ansi(s)?
        } else {
            Cow::Borrowed(s)
        };
        if self.s.wide && !is_printable_ascii(&t) {
            Ok(self.host.wcswidth(&t)?)
        } else {
            Ok(len_i(&t))
        }
    }

    /// The table's `width_fn` (`_multiline_width` when multiline: the widest line).
    pub fn width(&mut self, s: &str) -> Res<isize, H::Error> {
        if !self.s.is_multiline {
            return self.line_width(s);
        }
        let mut best = isize::MIN;
        for line in split_cr_lf(s) {
            best = best.max(self.line_width(line)?);
        }
        Ok(best)
    }

    /// The table's `width_fn` of an aligned cell (its padding included).
    pub fn aligned_width(&mut self, c: &Aligned<'_>) -> Res<isize, H::Error> {
        if is_printable_ascii(&c.text) {
            let n = c.left + c.right + c.text.len();
            return Ok(isize::try_from(n).unwrap_or(isize::MAX));
        }
        self.width(&c.padded())
    }

    fn afterpoint(&mut self, s: &str) -> Res<isize, H::Error> {
        let t = if self.s.has_invisible {
            self.strip_ansi(s)?
        } else {
            Cow::Borrowed(s)
        };
        match afterpoint(&t, self.s.max_digits, self.s.unicode_ok) {
            Some(n) => Ok(n),
            None => Ok(self.host.afterpoint(&t)?),
        }
    }

    /// `_align_column(strings, alignment, minwidth, ...)`.
    #[allow(clippy::too_many_lines)]
    pub fn align_column<'a>(
        &mut self,
        strings: Vec<Cow<'a, str>>,
        alignment: Align,
        minwidth: isize,
    ) -> Res<Vec<Aligned<'a>>, H::Error> {
        let strip = !self.s.preserve_whitespace;
        let mut extra = vec![0usize; strings.len()];
        let stripped = |v: Vec<Cow<'a, str>>| -> Vec<Cow<'a, str>> {
            if strip {
                v.into_iter().map(strip_cow).collect()
            } else {
                v
            }
        };
        let (strings, kind) = match alignment {
            Align::Right => (stripped(strings), Pad::Left),
            Align::Center => (stripped(strings), Pad::Both),
            Align::Decimal => {
                let mut decimals = Vec::with_capacity(strings.len());
                for s in &strings {
                    decimals.push(self.afterpoint(s)?);
                }
                let maxdecimals = decimals.iter().copied().max().ok_or(Error::EmptyMax)?;
                // `s + (maxdecimals - decs) * " "`: as a count for printable ASCII (widths
                // add up there), as text otherwise.
                let multiline = self.s.is_multiline;
                let out = strings
                    .into_iter()
                    .zip(&decimals)
                    .enumerate()
                    .map(|(i, (mut s, &d))| {
                        if maxdecimals > d {
                            if !multiline && is_printable_ascii(&s) {
                                extra[i] = (maxdecimals - d).unsigned_abs();
                            } else {
                                push_spaces(s.to_mut(), maxdecimals - d);
                            }
                        }
                        s
                    })
                    .collect();
                (out, Pad::Left)
            }
            Align::Falsy => (strings, Pad::None),
            Align::Left | Align::Other => (stripped(strings), Pad::Right),
        };
        let plain = !self.s.wide && !self.s.has_invisible;
        if self.s.is_multiline {
            let mut s_widths = Vec::with_capacity(strings.len());
            let mut maxwidth = isize::MIN;
            for ms in &strings {
                let mut ws = Vec::new();
                for line in split_cr_lf(ms) {
                    let w = self.line_width(line)?;
                    maxwidth = maxwidth.max(w);
                    ws.push(w);
                }
                s_widths.push(ws);
            }
            if strings.is_empty() {
                return Err(Error::EmptyMax);
            }
            let maxwidth = maxwidth.max(minwidth);
            let mut out = Vec::with_capacity(strings.len());
            for (ms, mw) in strings.iter().zip(&s_widths) {
                let lines = splitlines(ms);
                let mut parts = Vec::with_capacity(lines.len());
                if plain {
                    for line in lines {
                        parts.push(
                            pad(kind, maxwidth, Cow::Borrowed(line))?
                                .padded()
                                .into_owned(),
                        );
                    }
                } else {
                    let lens: Vec<isize> = split_cr_lf(ms).into_iter().map(len_i).collect();
                    let visible: Vec<isize> = mw
                        .iter()
                        .zip(&lens)
                        .map(|(w, l)| maxwidth - (w - l))
                        .collect();
                    for (line, w) in lines.into_iter().zip(visible) {
                        parts.push(pad(kind, w, Cow::Borrowed(line))?.padded().into_owned());
                    }
                }
                out.push(Aligned {
                    text: Cow::Owned(parts.join("\n")),
                    left: 0,
                    right: 0,
                });
            }
            return Ok(out);
        }
        let mut s_widths = Vec::with_capacity(strings.len());
        for (s, &x) in strings.iter().zip(&extra) {
            s_widths.push(self.line_width(s)? + isize::try_from(x).unwrap_or(isize::MAX));
        }
        let maxwidth = s_widths
            .iter()
            .copied()
            .max()
            .ok_or(Error::EmptyMax)?
            .max(minwidth);
        let mut out = Vec::with_capacity(strings.len());
        for ((s, w), x) in strings.into_iter().zip(&s_widths).zip(extra) {
            let xi = isize::try_from(x).unwrap_or(isize::MAX);
            let width = if plain {
                maxwidth
            } else {
                maxwidth - (w - (len_i(&s) + xi))
            };
            // `extra` spaces are part of the string tabulate pads.
            if width < 0 && kind != Pad::None {
                return Err(Error::NegativeWidth);
            }
            let mut a = pad(kind, (width - xi).max(0), s)?;
            a.right += x;
            out.push(a);
        }
        Ok(out)
    }

    /// `_align_header(header, alignment, width, visible_width, is_multiline, width_fn)`.
    pub fn align_header(
        &mut self,
        header: &str,
        alignment: Align,
        width: isize,
    ) -> Res<String, H::Error> {
        if self.s.is_multiline {
            let mut parts = Vec::new();
            for line in split_cr_lf(header) {
                let vw = self.width(line)?;
                parts.push(Self::align_header_line(line, alignment, width, vw)?);
            }
            return Ok(parts.join("\n"));
        }
        let vw = self.width(header)?;
        Self::align_header_line(header, alignment, width, vw)
    }

    fn align_header_line(
        header: &str,
        alignment: Align,
        width: isize,
        visible_width: isize,
    ) -> Res<String, H::Error> {
        let width = width + (len_i(header) - visible_width);
        let h = Cow::Borrowed(header);
        let out = match alignment {
            Align::Left => pad(Pad::Right, width, h)?,
            Align::Center => pad(Pad::Both, width, h)?,
            Align::Falsy => return Ok(header.to_owned()),
            _ => pad(Pad::Left, width, h)?,
        };
        Ok(out.padded().into_owned())
    }

    /// `_format_table(fmt, headers, headersaligns, rows, colwidths, colaligns, is_multiline,
    /// rowaligns)`. The rows are `cols[j][i]` in the order of `order`, which holds the
    /// re-inserted separators; `colaligns` and `headersaligns` are the lists the line and row
    /// producers see.
    #[allow(clippy::too_many_arguments)]
    pub fn format_table(
        &mut self,
        fmt: &TableFormat,
        headers: &[String],
        headersaligns: &[Align],
        cols: &[Vec<Aligned<'_>>],
        order: &[RowRef],
        colwidths: &[isize],
        colaligns: &[Align],
        rowaligns: &[RowAlign],
    ) -> Res<String, H::Error> {
        let total: usize = cols
            .iter()
            .map(|c| {
                c.iter()
                    .map(|s| s.text.len() + s.left + s.right + 3)
                    .sum::<usize>()
            })
            .sum();
        let mut out = Out {
            buf: String::with_capacity(total + 256),
            count: 0,
            bad: None,
        };
        let hidden = |i: usize| !headers.is_empty() && fmt.hide[i];
        let pad = fmt.padding;
        let padded_widths: Vec<isize> = colwidths.iter().map(|w| w + 2 * pad).collect();
        let cw = (&padded_widths[..], colaligns, AlignList::Columns);

        if fmt.lineabove.on() && !hidden(ABOVE) {
            self.emit_line(&mut out, cw, &fmt.lineabove)?;
        }
        if !headers.is_empty() {
            let hw = (&padded_widths[..], headersaligns, AlignList::Headers);
            self.emit_row(
                &mut out,
                Src::List(headers),
                hw,
                &fmt.headerrow,
                pad,
                RowAlign::Top,
            )?;
            if fmt.linebelowheader.on() && !hidden(BELOW_HEADER) {
                self.emit_line(&mut out, cw, &fmt.linebelowheader)?;
            }
        }
        let src = |r: RowRef| match r {
            RowRef::Data(i) => Src::Cols(cols, i),
            RowRef::Separator => Src::Separator,
        };
        if !order.is_empty() && fmt.linebetweenrows.on() && !hidden(BETWEEN) {
            let (&last, init) = order.split_last().expect("non-empty");
            // A plain line between rows is the same every time.
            let between = if let LineFmt::Plain { .. } = fmt.linebetweenrows {
                let mut tmp = Out {
                    buf: String::new(),
                    count: 0,
                    bad: None,
                };
                self.emit_line(&mut tmp, cw, &fmt.linebetweenrows)?;
                Some(tmp.buf)
            } else {
                None
            };
            for (&row, &ralign) in init.iter().zip(rowaligns) {
                if row != RowRef::Separator {
                    self.emit_row(&mut out, src(row), cw, &fmt.datarow, pad, ralign)?;
                }
                if let Some(line) = &between {
                    out.start();
                    out.buf.push_str(line);
                } else {
                    self.emit_line(&mut out, cw, &fmt.linebetweenrows)?;
                }
            }
            let ralign = *rowaligns.last().ok_or(Error::RowAlignIndex)?;
            self.emit_row(&mut out, src(last), cw, &fmt.datarow, pad, ralign)?;
        } else {
            let fallback = LineFmt::plain("", "", "", "");
            let separating_line = [
                &fmt.linebetweenrows,
                &fmt.linebelowheader,
                &fmt.linebelow,
                &fmt.lineabove,
            ]
            .into_iter()
            .find(|l| l.on())
            .unwrap_or(&fallback);
            for &row in order {
                match row {
                    RowRef::Separator => self.emit_line(&mut out, cw, separating_line)?,
                    RowRef::Data(_) => {
                        self.emit_row(&mut out, src(row), cw, &fmt.datarow, pad, RowAlign::Top)?;
                    }
                }
            }
        }
        if fmt.linebelow.on() && !hidden(BELOW) {
            self.emit_line(&mut out, cw, &fmt.linebelow)?;
        }
        if headers.is_empty() && order.is_empty() {
            return Ok(String::new()); // a completely empty table
        }
        if let Some((index, type_name)) = out.bad {
            return Err(Error::Join { index, type_name });
        }
        Ok(out.buf)
    }

    fn emit_line(
        &mut self,
        out: &mut Out,
        (widths, aligns, which): (&[isize], &[Align], AlignList),
        linefmt: &LineFmt,
    ) -> Res<(), H::Error> {
        if let LineFmt::Plain {
            begin,
            fill,
            sep,
            end,
        } = linefmt
        {
            let start = out.start();
            out.buf.push_str(begin);
            for (j, &w) in widths.iter().enumerate() {
                if j > 0 {
                    out.buf.push_str(sep);
                }
                if let Ok(n) = usize::try_from(w) {
                    let mut chars = fill.chars();
                    match (chars.next(), chars.next()) {
                        (Some(c), None) => out.buf.extend(std::iter::repeat_n(c, n)),
                        _ => {
                            for _ in 0..n {
                                out.buf.push_str(fill);
                            }
                        }
                    }
                }
            }
            out.buf.push_str(end);
            out.rstrip_from(start);
            return Ok(());
        }
        let p = self.build_line(widths, aligns, which, linefmt)?;
        out.piece(p);
        Ok(())
    }

    fn emit_row(
        &mut self,
        out: &mut Out,
        cells: Src<'_>,
        (padded_widths, aligns, which): (&[isize], &[Align], AlignList),
        rowfmt: &RowFmt,
        pad: isize,
        rowalign: RowAlign,
    ) -> Res<(), H::Error> {
        if !self.s.is_multiline {
            if let RowFmt::Plain { begin, sep, end } = rowfmt {
                // _build_simple_row(_pad_row(cells, pad), rowfmt); the separator is not padded.
                let start = out.start();
                out.buf.push_str(begin);
                if let Src::Separator = cells {
                    out.buf.push('\u{1}');
                } else {
                    for j in 0..cells.len() {
                        if j > 0 {
                            out.buf.push_str(sep);
                        }
                        cells.write(j, &mut out.buf, pad);
                    }
                }
                out.buf.push_str(end);
                out.rstrip_from(start);
                return Ok(());
            }
            if let RowFmt::Html { th, unsafe_ } = rowfmt {
                emit_html_row(out, cells, aligns, *th, *unsafe_, pad);
                return Ok(());
            }
            if let RowFmt::Latex { raw } = rowfmt {
                let start = out.start();
                for j in 0..cells.len() {
                    if j > 0 {
                        out.buf.push('&');
                    }
                    let (l, text, r) = cells.parts(j, pad);
                    out.buf.extend(std::iter::repeat_n(' ', l));
                    if *raw {
                        out.buf.push_str(text);
                    } else {
                        latex_escape_into(&mut out.buf, text);
                    }
                    out.buf.extend(std::iter::repeat_n(' ', r));
                }
                out.buf.push_str("\\\\");
                out.rstrip_from(start);
                return Ok(());
            }
            let p = if let Src::Separator = cells {
                self.build_row(Cells::Separator, padded_widths, aligns, which, rowfmt)?
            } else {
                let padded = cells.padded(pad);
                self.build_row(Cells::List(&padded), padded_widths, aligns, which, rowfmt)?
            };
            out.piece(p);
            return Ok(());
        }
        // _append_multiline_row: the producers see the unpadded widths.
        let colwidths: Vec<isize> = padded_widths.iter().map(|w| w - 2 * pad).collect();
        let texts: Vec<Cow<'_, str>> = (0..cells.len()).map(|j| cells.get(j)).collect();
        let cells_lines: Vec<Vec<&str>> = texts.iter().map(|c| splitlines(c)).collect();
        let nlines = cells_lines
            .iter()
            .map(Vec::len)
            .max()
            .ok_or(Error::EmptyMax)?;
        let aligned: Vec<Vec<Cow<'_, str>>> = cells_lines
            .into_iter()
            .zip(&colwidths)
            .map(|(cl, &w)| align_cell_vertically(cl, nlines, w, rowalign))
            .collect();
        for i in 0..nlines {
            let ln: Vec<String> = aligned.iter().map(|cl| cl[i].to_string()).collect();
            let padded = pad_cells(&ln, pad);
            let p = self.build_row(Cells::List(&padded), &colwidths, aligns, which, rowfmt)?;
            out.piece(p);
        }
        Ok(())
    }

    fn build_line(
        &mut self,
        colwidths: &[isize],
        aligns: &[Align],
        which: AlignList,
        linefmt: &LineFmt,
    ) -> Res<Piece, H::Error> {
        let s = match linefmt {
            LineFmt::Off => return Ok(Piece::Other("NoneType".into())),
            LineFmt::Plain {
                begin,
                fill,
                sep,
                end,
            } => {
                let cells: Vec<String> =
                    colwidths.iter().map(|&w| pystr::repeat(fill, w)).collect();
                simple_row(begin, sep, end, cells.iter().map(String::as_str))
            }
            LineFmt::PipeColons => {
                let segs: Vec<String> = if aligns.is_empty() {
                    colwidths
                        .iter()
                        .map(|&w| pipe_segment(Align::Falsy, w))
                        .collect()
                } else {
                    aligns
                        .iter()
                        .zip(colwidths)
                        .map(|(&a, &w)| pipe_segment(a, w))
                        .collect()
                };
                format!("|{}|", segs.join("|"))
            }
            LineFmt::GridColons => {
                let segs: Vec<String> = if aligns.is_empty() {
                    colwidths
                        .iter()
                        .map(|&w| grid_segment(w, Align::Falsy))
                        .collect()
                } else {
                    aligns
                        .iter()
                        .zip(colwidths)
                        .map(|(&a, &w)| grid_segment(w, a))
                        .collect()
                };
                format!("+{}+", segs.join("+"))
            }
            LineFmt::HtmlBegin => "<table>\n<tbody>".to_owned(),
            LineFmt::LatexBegin {
                booktabs,
                longtable,
            } => {
                let cols: String = aligns
                    .iter()
                    .map(|a| match a {
                        Align::Right | Align::Decimal => 'r',
                        Align::Center => 'c',
                        _ => 'l',
                    })
                    .collect();
                format!(
                    "{}{}}}\n{}",
                    if *longtable {
                        "\\begin{longtable}{"
                    } else {
                        "\\begin{tabular}{"
                    },
                    cols,
                    if *booktabs { "\\toprule" } else { "\\hline" }
                )
            }
            LineFmt::Asciidoc => asciidoc_header_line(false, colwidths, aligns, which)?,
            LineFmt::Custom(id) => return Ok(self.host.line(*id, colwidths, which)?),
        };
        Ok(Piece::Str(s))
    }

    #[allow(clippy::too_many_lines)]
    fn build_row(
        &mut self,
        cells: Cells<'_>,
        colwidths: &[isize],
        aligns: &[Align],
        which: AlignList,
        rowfmt: &RowFmt,
    ) -> Res<Piece, H::Error> {
        const SEP: &str = "\u{1}";
        let sep_cells = [SEP.to_owned()];
        let list: &[String] = match cells {
            Cells::List(l) => l,
            Cells::Separator => &sep_cells,
        };
        let s = match rowfmt {
            RowFmt::Off => return Ok(Piece::Other("NoneType".into())),
            RowFmt::Plain { begin, sep, end } => {
                simple_row(begin, sep, end, list.iter().map(String::as_str))
            }
            RowFmt::Mediawiki(sep) => {
                let values: Vec<String> = list
                    .iter()
                    .zip(aligns)
                    .map(|(c, a)| {
                        let attr = match a {
                            Align::Right | Align::Decimal => "style=\"text-align: right;\"| ",
                            Align::Center => "style=\"text-align: center;\"| ",
                            _ => "",
                        };
                        format!(" {attr}{c} ")
                    })
                    .collect();
                let colsep: String = [*sep, *sep].iter().collect();
                let mut row = sep.to_string();
                row.push_str(&values.join(&colsep));
                rstrip(&row).to_owned()
            }
            RowFmt::Moin { header } => {
                let h = if *header { "'''" } else { "" };
                let mut row = String::new();
                for (c, a) in list.iter().zip(aligns) {
                    let attr = match a {
                        Align::Right | Align::Decimal => "<style=\"text-align: right;\">",
                        Align::Center => "<style=\"text-align: center;\">",
                        _ => "",
                    };
                    row.push_str("||");
                    row.push_str(attr);
                    row.push(' ');
                    row.push_str(h);
                    row.push_str(c);
                    row.push_str(h);
                    row.push(' ');
                }
                row.push_str("||");
                row
            }
            RowFmt::Html { th, unsafe_ } => {
                let tag = if *th { "th" } else { "td" };
                let mut body = String::new();
                for (c, a) in list.iter().zip(aligns) {
                    let attr = match a {
                        Align::Right | Align::Decimal => " style=\"text-align: right;\"",
                        Align::Center => " style=\"text-align: center;\"",
                        _ => "",
                    };
                    body.push('<');
                    body.push_str(tag);
                    body.push_str(attr);
                    body.push('>');
                    if *unsafe_ {
                        body.push_str(c);
                    } else {
                        html_escape(&mut body, c);
                    }
                    body.push_str("</");
                    body.push_str(tag);
                    body.push('>');
                }
                let row = format!("<tr>{}</tr>", rstrip(&body));
                if *th {
                    format!("<table>\n<thead>\n{row}\n</thead>\n<tbody>")
                } else {
                    row
                }
            }
            RowFmt::Latex { raw } => {
                let escaped: Vec<String> = list
                    .iter()
                    .map(|c| if *raw { c.clone() } else { latex_escape(c) })
                    .collect();
                simple_row("", "&", "\\\\", escaped.iter().map(String::as_str))
            }
            RowFmt::Textile => {
                let Cells::List(list) = cells else {
                    return Err(Error::TextileSeparator);
                };
                let (first, rest) = list.split_first().ok_or(Error::TextileEmpty)?;
                let first = format!("{first} ");
                let mut parts = Vec::with_capacity(list.len());
                for (v, a) in std::iter::once(&first).chain(rest).zip(aligns) {
                    let attr = match a {
                        Align::Left => "<.",
                        Align::Right | Align::Decimal => ">.",
                        Align::Center => "=.",
                        _ => "",
                    };
                    parts.push(format!("{attr}{v}"));
                }
                format!("|{}|", parts.join("|"))
            }
            RowFmt::Asciidoc { header } => {
                let data = format!("|{}", list.join("|"));
                if *header {
                    format!(
                        "{}\n{data}",
                        asciidoc_header_line(true, colwidths, aligns, which)?
                    )
                } else {
                    data
                }
            }
            RowFmt::Custom(id) => return Ok(self.host.row(*id, cells, colwidths, which)?),
        };
        Ok(Piece::Str(s))
    }
}

/// `_align_cell_veritically(text_lines, num_lines, column_width, row_alignment)`.
pub fn align_cell_vertically(
    lines: Vec<&str>,
    num_lines: usize,
    width: isize,
    align: RowAlign,
) -> Vec<Cow<'_, str>> {
    let delta = num_lines.saturating_sub(lines.len());
    let blank = || Cow::Owned(pystr::repeat(" ", width));
    let (top, bottom) = match align {
        RowAlign::Bottom => (delta, 0),
        RowAlign::Center => (delta / 2, delta - delta / 2),
        RowAlign::Top => (0, delta),
    };
    let mut out = Vec::with_capacity(num_lines);
    out.extend((0..top).map(|_| blank()));
    out.extend(lines.into_iter().map(Cow::Borrowed));
    out.extend((0..bottom).map(|_| blank()));
    out
}

/// `_pad_row(cells, padding)` for a list of cells.
fn pad_cells(cells: &[String], padding: isize) -> Vec<String> {
    if padding <= 0 {
        return cells.to_vec();
    }
    cells
        .iter()
        .map(|c| {
            let mut s = String::with_capacity(c.len() + 2 * padding.unsigned_abs());
            push_spaces(&mut s, padding);
            s.push_str(c);
            push_spaces(&mut s, padding);
            s
        })
        .collect()
}

/// `_build_simple_row`: `(begin + sep.join(cells) + end).rstrip()`.
fn simple_row<'a>(
    begin: &str,
    sep: &str,
    end: &str,
    cells: impl Iterator<Item = &'a str>,
) -> String {
    let mut s = String::from(begin);
    for (i, c) in cells.enumerate() {
        if i > 0 {
            s.push_str(sep);
        }
        s.push_str(c);
    }
    s.push_str(end);
    let keep = rstrip(&s).len();
    s.truncate(keep);
    s
}

fn pipe_segment(align: Align, w: isize) -> String {
    match align {
        Align::Right | Align::Decimal => pystr::repeat("-", w - 1) + ":",
        Align::Center => format!(":{}:", pystr::repeat("-", w - 2)),
        Align::Left => format!(":{}", pystr::repeat("-", w - 1)),
        _ => pystr::repeat("-", w),
    }
}

fn grid_segment(w: isize, align: Align) -> String {
    match align {
        Align::Right => pystr::repeat("=", w - 1) + ":",
        Align::Center => format!(":{}:", pystr::repeat("=", w - 2)),
        Align::Left => format!(":{}", pystr::repeat("=", w - 1)),
        _ => pystr::repeat("=", w),
    }
}

fn asciidoc_header_line<E>(
    is_header: bool,
    colwidths: &[isize],
    aligns: &[Align],
    which: AlignList,
) -> Res<String, E> {
    let mut marks = Vec::with_capacity(aligns.len());
    for (i, a) in aligns.iter().enumerate() {
        marks.push(match a {
            Align::Left => '<',
            Align::Right | Align::Decimal => '>',
            Align::Center => '^',
            _ => {
                return Err(Error::AsciidocKey {
                    list: which,
                    index: i,
                });
            }
        });
    }
    let specs: Vec<String> = colwidths
        .iter()
        .zip(&marks)
        .map(|(w, c)| format!("{c}{w}"))
        .collect();
    let mut header = format!("cols=\"{}\"", specs.join(","));
    if is_header {
        header.push_str(",options=\"header\"");
    }
    Ok(format!("[{header}]\n|===="))
}

/// `_html_row_with_attrs(celltag, unsafe, _pad_row(cells, pad), colwidths, colaligns)`.
fn emit_html_row(
    out: &mut Out,
    cells: Src<'_>,
    aligns: &[Align],
    th: bool,
    unsafe_: bool,
    pad: isize,
) {
    let tag = if th { "th" } else { "td" };
    let start = out.start();
    let _ = start;
    if th {
        out.buf.push_str("<table>\n<thead>\n");
    }
    out.buf.push_str("<tr>");
    let cells_start = out.buf.len();
    for (j, a) in (0..cells.len()).zip(aligns) {
        let attr = match a {
            Align::Right | Align::Decimal => " style=\"text-align: right;\"",
            Align::Center => " style=\"text-align: center;\"",
            _ => "",
        };
        out.buf.push('<');
        out.buf.push_str(tag);
        out.buf.push_str(attr);
        out.buf.push('>');
        let (l, text, r) = cells.parts(j, pad);
        out.buf.extend(std::iter::repeat_n(' ', l));
        if unsafe_ {
            out.buf.push_str(text);
        } else {
            html_escape(&mut out.buf, text);
        }
        out.buf.extend(std::iter::repeat_n(' ', r));
        out.buf.push_str("</");
        out.buf.push_str(tag);
        out.buf.push('>');
    }
    out.rstrip_from(cells_start);
    out.buf.push_str("</tr>");
    if th {
        out.buf.push_str("\n</thead>\n<tbody>");
    }
}

/// `html.escape(s)` (quote=True).
fn html_escape(out: &mut String, s: &str) {
    if !s
        .bytes()
        .any(|b| matches!(b, b'&' | b'<' | b'>' | b'"' | b'\''))
    {
        out.push_str(s);
        return;
    }
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            c => out.push(c),
        }
    }
}

/// `_latex_row`'s per-char escapes (`LATEX_ESCAPE_RULES`).
fn latex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    latex_escape_into(&mut out, s);
    out
}

fn latex_escape_into(out: &mut String, s: &str) {
    if !s.bytes().any(|b| {
        matches!(
            b,
            b'&' | b'%' | b'$' | b'#' | b'_' | b'^' | b'{' | b'}' | b'~' | b'\\' | b'<' | b'>'
        )
    }) {
        out.push_str(s);
        return;
    }
    for c in s.chars() {
        match c {
            '&' => out.push_str("\\&"),
            '%' => out.push_str("\\%"),
            '$' => out.push_str("\\$"),
            '#' => out.push_str("\\#"),
            '_' => out.push_str("\\_"),
            '^' => out.push_str("\\^{}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '\\' => out.push_str("\\textbackslash{}"),
            '<' => out.push_str("\\ensuremath{<}"),
            '>' => out.push_str("\\ensuremath{>}"),
            c => out.push(c),
        }
    }
}
