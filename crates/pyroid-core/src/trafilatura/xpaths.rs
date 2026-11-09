// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! Trafilatura's XPath expressions (xpaths.py, settings.BASIC_CLEAN_XPATH, external's
//! SANITIZED_XPATH and the few inline ones), hand-compiled to tree walks.
//!
//! libxml2 semantics reproduced here (xpath.c 2.14.6, lxml 6.1.3):
//! - `.//*[p]` selects every descendant element (not the context node) satisfying `p`, in
//!   document order; the result is computed before Trafilatura deletes anything.
//! - `.//*[p][1]` keeps, per parent, the first child satisfying `p`; the first such node in
//!   document order is the first descendant satisfying `p`, which is all `next(expr(tree))`
//!   looks at.
//! - `string(@id|@class)`, as a function argument, is the string value of the first node of the
//!   union in document order: for two attributes of one element, whichever comes first in its
//!   attribute list. lxml hands EXSLT `re:test` the same first node.
//! - a missing attribute is the empty string; `@x='v'` is true iff the attribute exists with
//!   value v.
//! - `re:test(s, pat[, 'i'])` is Python `re.search(pat, s[, re.IGNORECASE])`, see [`matchers`].

use super::matchers as m;
use crate::html::tree::{Document, Kind, Matcher, NIL, Name, NodeId, names as t};

/// `string(@name)`.
#[inline]
pub fn attr(d: &Document, id: NodeId, name: Name) -> &str {
    d.attr(id, name).unwrap_or("")
}

/// `string(@a|@b)`: the value of whichever of the two attributes comes first, "" if neither.
pub fn first_attr(d: &Document, id: NodeId, a: Name, b: Name) -> &str {
    d.n(id)
        .attrs
        .iter()
        .find(|x| x.name == a || x.name == b)
        .map_or("", |x| x.value.as_str())
}

fn attr_eq(d: &Document, id: NodeId, name: Name, v: &str) -> bool {
    d.attr(id, name) == Some(v)
}

/// `contains(translate(s, from, to), needle)` with single-character maps.
fn contains_translated(s: &str, from: &[char], to: &[char], needle: &str) -> bool {
    if !s.chars().any(|c| from.contains(&c)) {
        return s.contains(needle);
    }
    let t: String = s
        .chars()
        .map(|c| from.iter().position(|&f| f == c).map_or(c, |i| to[i]))
        .collect();
    t.contains(needle)
}

/// `re:test(s, pattern)`; Python's `re.search` on "" never matches these patterns.
#[inline]
fn re_test(set: &m::LitSet, s: &str) -> bool {
    !s.is_empty() && set.search(s)
}

#[inline]
fn tag_in(d: &Document, id: NodeId, tags: &[Name]) -> bool {
    let n = d.n(id);
    n.kind == Kind::Element && tags.contains(&n.tag)
}

/// `.//*[pred]`: descendant elements of `ctx` satisfying `pred`, document order.
pub fn select(d: &Document, ctx: NodeId, pred: impl Fn(&Document, NodeId) -> bool) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut c = d.traverse_next(ctx, ctx);
    while c != NIL {
        if d.is_element(c) && pred(d, c) {
            out.push(c);
        }
        c = d.traverse_next(ctx, c);
    }
    out
}

/// First descendant satisfying `pred` (`(.//*[pred])[1]`).
pub fn select_first(
    d: &Document,
    ctx: NodeId,
    pred: impl Fn(&Document, NodeId) -> bool,
) -> Option<NodeId> {
    let mut c = d.traverse_next(ctx, ctx);
    while c != NIL {
        if d.is_element(c) && pred(d, c) {
            return Some(c);
        }
        c = d.traverse_next(ctx, c);
    }
    None
}

fn has_descendant_tag(d: &Document, id: NodeId, tag: Name) -> bool {
    select_first(d, id, |d, c| d.tag(c) == tag).is_some()
}

const DIV_SECTION_ASIDE: &[Name] = &[t::DIV, t::SECTION, t::ASIDE];
const COMMENT_CONTAINERS: &[Name] = &[t::DIV, t::LIST, t::SECTION, t::DETAILS];
const COMMENT_CONTAINERS_LISTS: &[Name] = &[t::DIV, t::LIST, t::SECTION, t::DETAILS, t::OL, t::UL];
const DISCARD_SCOPE: &[Name] = &[t::DIV, t::ITEM, t::LIST, t::P, t::SECTION, t::SPAN];
const BODY_SCOPE: &[Name] = &[t::ARTICLE, t::DIV, t::MAIN, t::SECTION];
const DIV_LIST_SECTION: &[Name] = &[t::DIV, t::LIST, t::SECTION];
const DIV_SECTION: &[Name] = &[t::DIV, t::SECTION];

/// A list of XPath expressions, each a selector over the context node.
pub type XList = &'static [fn(&Document, NodeId) -> Vec<NodeId>];

pub static RAW_TREE_PRUNE_XPATH: XList = &[
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DIV_SECTION_ASIDE)
                && (re_test(&m::MVP_OR_SCROLL, attr(d, e, t::ID))
                    || re_test(&m::MVP_OR_SCROLL, attr(d, e, t::CLASS)))
                && (!has_descendant_tag(d, e, t::H1)
                    || re_test(&m::MVP, attr(d, e, t::ID))
                    || re_test(&m::MVP, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DIV_SECTION_ASIDE)
                && attr(d, e, t::CLASS).contains("elementor-share-buttons")
        })
    },
];

pub static REMOVE_COMMENTS_AND_LISTS_XPATH: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, COMMENT_CONTAINERS_LISTS)
            && (m::comments_id(attr(d, e, t::ID)) || m::comments_class(attr(d, e, t::CLASS)))
            && (!tag_in(d, e, &[t::OL, t::UL])
                || m::comments_id_short(attr(d, e, t::ID))
                || m::comments_class(attr(d, e, t::CLASS)))
    })
}];

pub static REMOVE_COMMENTS_XPATH: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, COMMENT_CONTAINERS)
            && (m::comments_id(attr(d, e, t::ID)) || m::comments_class(attr(d, e, t::CLASS)))
    })
}];

pub static OVERALL_DISCARD_XPATH: XList = &[
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DISCARD_SCOPE)
                && (d.get(e, "data-lp-replacement-content").is_some()
                    || contains_translated(attr(d, e, t::ROLE), &['N'], &['n'], "nav")
                    || d.get(e, "data-component")
                        .unwrap_or("")
                        .contains("MostPopularStories")
                    || first_attr(d, e, t::ID, t::CLASS).contains("cookie")
                    || re_test(&m::OD_ID, attr(d, e, t::ID))
                    || re_test(&m::OD_CLASS, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| {
        select(d, ctx, |d, e| {
            attr_eq(d, e, t::CLASS, "comments-title")
                || first_attr(d, e, t::ID, t::CLASS).starts_with("reply-")
                || first_attr(d, e, t::ID, t::STYLE).contains("hidden")
                || attr(d, e, t::STYLE).contains("display:none")
                || attr(d, e, t::STYLE).contains("display: none")
                || re_test(&m::OD1_ID, attr(d, e, t::ID))
                || re_test(&m::OD1_CLASS, attr(d, e, t::CLASS))
                || attr_eq(d, e, t::ARIA_HIDDEN, "true")
        })
    },
];

pub static TEASER_DISCARD_XPATH: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, DISCARD_SCOPE)
            && (contains_translated(attr(d, e, t::ID), &['T'], &['t'], "teaser")
                || contains_translated(attr(d, e, t::CLASS), &['T'], &['t'], "teaser"))
    })
}];

pub static PRECISION_DISCARD_XPATH: XList = &[
    |d, ctx| select(d, ctx, |d, e| d.tag(e) == t::HEADER),
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DISCARD_SCOPE) && {
                let s = first_attr(d, e, t::ID, t::CLASS);
                m::bottom(s) || m::link_word(s) || attr(d, e, t::STYLE).contains("border")
            }
        })
    },
];

pub static DISCARD_IMAGE_ELEMENTS: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, DISCARD_SCOPE)
            && (attr(d, e, t::ID).contains("caption") || attr(d, e, t::CLASS).contains("caption"))
    })
}];

pub static COMMENTS_DISCARD_XPATH: XList = &[
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DIV_SECTION) && attr(d, e, t::ID).starts_with("respond")
        })
    },
    |d, ctx| select(d, ctx, |d, e| tag_in(d, e, &[t::CITE, t::QUOTE])),
    |d, ctx| {
        select(d, ctx, |d, e| {
            attr_eq(d, e, t::CLASS, "comments-title")
                || attr(d, e, t::STYLE).contains("display:none")
                || re_test(&m::COMMENTS_DISCARD_CLASS, attr(d, e, t::CLASS))
                || re_test(
                    &m::COMMENTS_DISCARD_IDCLASS,
                    first_attr(d, e, t::ID, t::CLASS),
                )
        })
    },
];

/// One expression returning the first node of its result, if any.
pub type XFirst = fn(&Document, NodeId) -> Option<NodeId>;

pub static BODY_XPATH: &[XFirst] = &[
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, BODY_SCOPE)
                && (attr_eq(d, e, t::CLASS, "post")
                    || attr_eq(d, e, t::CLASS, "entry")
                    || attr_eq(d, e, t::ITEMPROP, "articleBody")
                    || attr_eq(d, e, t::ID, "articleContent")
                    || re_test(&m::BODY0_ID, attr(d, e, t::ID))
                    || re_test(&m::BODY0_CLASS, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| select_first(d, ctx, |d, e| d.tag(e) == t::ARTICLE),
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, BODY_SCOPE)
                && (attr_eq(d, e, t::ROLE, "article")
                    || attr_eq(d, e, t::ID, "article")
                    || attr_eq(d, e, t::ID, "story")
                    || ["postarea", "art-postcontent", "text", "cell", "story"]
                        .iter()
                        .any(|v| attr_eq(d, e, t::CLASS, v))
                    || re_test(&m::BODY2_ID, attr(d, e, t::ID))
                    || m::BODY2_FULLTEXT.search_icase(attr(d, e, t::CLASS))
                    || re_test(&m::BODY2_CLASS, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, BODY_SCOPE)
                && (attr_eq(d, e, t::ID, "content")
                    || attr_eq(d, e, t::CLASS, "content")
                    || re_test(&m::BODY3_ID, attr(d, e, t::ID))
                    || re_test(&m::BODY3_CLASS, attr(d, e, t::CLASS))
                    || contains_translated(
                        attr(d, e, t::ID),
                        &['C', 'M'],
                        &['c', 'm'],
                        "main-content",
                    )
                    || contains_translated(
                        attr(d, e, t::CLASS),
                        &['C', 'M'],
                        &['c', 'm'],
                        "main-content",
                    )
                    || contains_translated(
                        attr(d, e, t::CLASS),
                        &['C', 'P'],
                        &['c', 'p'],
                        "page-content",
                    ))
        })
    },
    |d, ctx| {
        // (.//*[self::article or self::div or self::section][starts-with(...)])[1]|(.//main)[1]
        let a = select_first(d, ctx, |d, e| {
            tag_in(d, e, &[t::ARTICLE, t::DIV, t::SECTION])
                && (attr(d, e, t::CLASS).starts_with("main")
                    || attr(d, e, t::ID).starts_with("main")
                    || attr(d, e, t::ROLE).starts_with("main"))
        });
        let b = select_first(d, ctx, |d, e| d.tag(e) == t::MAIN);
        match (a, b) {
            (Some(a), Some(b)) => Some(if precedes(d, ctx, a, b) { a } else { b }),
            (a, b) => a.or(b),
        }
    },
];

/// COMMENTS_XPATH.
pub static COMMENTS_XPATH: &[XFirst] = &[
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, DIV_LIST_SECTION)
                && (re_test(&m::COMMENTS0_IDCLASS, first_attr(d, e, t::ID, t::CLASS))
                    || re_test(&m::COMMENTS0_CLASS, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, DIV_LIST_SECTION)
                && (re_test(&m::COMMENTS1_IDCLASS, first_attr(d, e, t::ID, t::CLASS))
                    || re_test(&m::COMMENTS1_CLASS, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, DIV_LIST_SECTION) && re_test(&m::COMMENTS2_ID, attr(d, e, t::ID))
        })
    },
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, DIV_SECTION)
                && (attr(d, e, t::ID).starts_with("social")
                    || attr(d, e, t::CLASS).contains("comment"))
        })
    },
];

/// Whether `a` comes before `b` in document order below `ctx` (both descendants of it).
fn precedes(d: &Document, ctx: NodeId, a: NodeId, b: NodeId) -> bool {
    if a == b {
        return false;
    }
    let mut c = d.traverse_next(ctx, ctx);
    while c != NIL {
        if c == a {
            return true;
        }
        if c == b {
            return false;
        }
        c = d.traverse_next(ctx, c);
    }
    false
}

/// recover_wild_text: `.//code|.//p|.//quote|.//table|.//div[contains(@class, 'w3-code')]`,
/// plus `|.//div|.//lb|.//list` in recall mode.
pub fn wild_text_search(d: &Document, ctx: NodeId, recall: bool) -> Vec<NodeId> {
    select(d, ctx, |d, e| {
        let g = d.tag(e);
        g == t::CODE
            || g == t::P
            || g == t::QUOTE
            || g == t::TABLE
            || (g == t::DIV && (recall || attr(d, e, t::CLASS).contains("w3-code")))
            || (recall && (g == t::LB || g == t::LIST))
    })
}

/// convert_quotes: `.//span[starts-with(@class,'hljs')]`.
pub fn hljs_spans(d: &Document, ctx: NodeId) -> Vec<NodeId> {
    select(d, ctx, |d, e| {
        d.tag(e) == t::SPAN && attr(d, e, t::CLASS).starts_with("hljs")
    })
}

/// recover_math: `.//*[local-name()="annotation"][@encoding="application/x-tex"]` (HTML trees
/// have no namespaces).
pub fn tex_annotations(d: &Document, ctx: NodeId) -> Vec<NodeId> {
    select(d, ctx, |d, e| {
        d.tag(e) == t::ANNOTATION && (d.get(e, "encoding") == Some("application/x-tex"))
    })
}

/// settings.BASIC_CLEAN_XPATH.
pub fn basic_clean(d: &Document, ctx: NodeId) -> Vec<NodeId> {
    select(d, ctx, |d, e| {
        let g = d.tag(e);
        g == t::ASIDE
            || (g == t::DIV && first_attr(d, e, t::CLASS, t::ID).contains("footer"))
            || g == t::FENCEDFRAME
            || g == t::FOOTER
            || g == t::SCRIPT
            || g == t::STYLE
            || g == t::SVG
            || g == t::TEMPLATE
            || d.attr(e, t::CLASS)
                .is_some_and(|s| m::COOKIE_CONSENT.search_icase(s))
            || d.attr(e, t::ID)
                .is_some_and(|s| m::COOKIE_CONSENT.search_icase(s))
    })
}

/// external.SANITIZED_XPATH is non-empty.
pub fn sanitized_any(d: &Document, body: NodeId) -> bool {
    const TAGS: &[Name] = &[
        t::ASIDE,
        t::AUDIO,
        t::BUTTON,
        t::FENCEDFRAME,
        t::FIELDSET,
        t::FIGURE,
        t::FOOTER,
        t::IFRAME,
        t::INPUT,
        t::LABEL,
        t::LINK,
        t::NAV,
        t::NOINDEX,
        t::NOSCRIPT,
        t::OBJECT,
        t::OPTION,
        t::SELECT,
        t::SOURCE,
        t::SVG,
        t::TIME,
    ];
    select_first(d, body, |d, e| TAGS.contains(&d.tag(e))).is_some()
}

/// `//p//text()` evaluated in the document whose root element is `root`: whether any text
/// node exists, and their total length in code points. lxml keeps a removed element in its
/// document, so `//` from a detached context node still means the original document.
pub fn p_text(d: &Document, root: NodeId) -> (bool, usize) {
    let mut any = false;
    let mut len = 0usize;
    // (node, inside a p) in document order, iteratively
    let mut stack: Vec<(NodeId, bool)> = vec![(root, false)];
    while let Some((id, in_p)) = stack.pop() {
        let n = d.n(id);
        let here = in_p || (n.kind == Kind::Element && n.tag == t::P);
        if n.kind == Kind::Element && here {
            if let Some(s) = &n.text {
                any = true;
                len += super::pystr::py_len(s);
            }
        }
        if here {
            for c in d.children(id) {
                if let Some(s) = d.tail(c) {
                    any = true;
                    len += super::pystr::py_len(s);
                }
            }
        }
        let kids: Vec<NodeId> = d.children(id).filter(|&c| d.is_element(c)).collect();
        for &c in kids.iter().rev() {
            stack.push((c, here));
        }
    }
    (any, len)
}

/// `body.xpath(".//p//text()")` is non-empty: a descendant <p> holds a text node (even "").
pub fn p_text_rel(d: &Document, body: NodeId) -> bool {
    d.iter_vec(body, &Matcher::tags(&[t::P]), false)
        .into_iter()
        .any(|p| {
            let mut n = 0;
            d.itertext(p, &mut |_| n += 1);
            n > 0
        })
}
