// Derived from trafilatura 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
//! trafilatura 1.11.0's xpaths.py, hand-compiled like [`crate::trafilatura::xpaths`]. These
//! use only `contains`, `starts-with`, `translate` and equality, so each attribute test is a
//! literal set (a `^` prefix marks `starts-with`).

use std::sync::LazyLock;

use super::super::matchers::LitSet;
use super::super::xpaths::{XFirst, XList, attr, select, select_first};
use crate::html::tree::{Document, Kind, Name, NodeId, names as t};

macro_rules! litset {
    ($name:ident, [$($l:literal),* $(,)?]) => {
        static $name: LazyLock<LitSet> = LazyLock::new(|| LitSet::new(&[$($l),*]));
    };
}

#[inline]
fn tag_in(d: &Document, id: NodeId, tags: &[Name]) -> bool {
    let n = d.n(id);
    n.kind == Kind::Element && tags.contains(&n.tag)
}

#[inline]
fn has(set: &LitSet, s: &str) -> bool {
    !s.is_empty() && set.search(s)
}

fn eq(d: &Document, id: NodeId, name: Name, v: &str) -> bool {
    d.attr(id, name) == Some(v)
}

/// `contains(translate(s, X, x), needle)` for one upper-case letter.
fn contains_lowered(s: &str, upper: char, needle: &str) -> bool {
    if s.contains(upper) {
        let lower = upper.to_ascii_lowercase();
        s.chars()
            .map(|c| if c == upper { lower } else { c })
            .collect::<String>()
            .contains(needle)
    } else {
        s.contains(needle)
    }
}

const ARTICLE_DIV_MAIN_SECTION: &[Name] = &[t::ARTICLE, t::DIV, t::MAIN, t::SECTION];
const DIV_LIST_SECTION: &[Name] = &[t::DIV, t::LIST, t::SECTION];
const DIV_SECTION: &[Name] = &[t::DIV, t::SECTION];
const DISCARD_SCOPE: &[Name] = &[t::DIV, t::ITEM, t::LIST, t::P, t::SECTION, t::SPAN];

litset!(
    BODY0_CLASS,
    [
        "post-text",
        "post_text",
        "post-body",
        "post-entry",
        "postentry",
        "post-content",
        "post_content",
        "postcontent",
        "postContent",
        "post_inner_wrapper",
        "article-text",
        "articletext",
        "articleText",
        "entry-content",
        "article-content",
        "article__content",
        "article-body",
        "article__body",
        "ArticleContent",
        "page-content",
        "text-content",
        "body-text",
        "article__container",
        "art-content",
    ]
);
litset!(
    BODY0_ID,
    [
        "entry-content",
        "article-content",
        "article__content",
        "article-body",
        "article__body",
        "body-text",
        "art-content",
    ]
);
litset!(
    BODY2_CLASS,
    [
        "post-bodycopy",
        "storycontent",
        "story-content",
        "theme-content",
        "blog-content",
        "section-content",
        "single-content",
        "single-post",
        "main-column",
        "wpb_text_column",
        "^article ",
        "story-body",
        "field-body",
    ]
);
litset!(BODY2_ID, ["^primary", "story-body"]);
litset!(BODY3_ID, ["content-main", "content-body", "contentBody"]);
litset!(
    BODY3_CLASS,
    [
        "content-main",
        "content_main",
        "content-body",
        "content__body"
    ]
);

/// BODY_XPATH.
pub static BODY_XPATH: &[XFirst] = &[
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, ARTICLE_DIV_MAIN_SECTION) && {
                let (c, i) = (attr(d, e, t::CLASS), attr(d, e, t::ID));
                c == "post"
                    || c == "entry"
                    || has(&BODY0_CLASS, c)
                    || has(&BODY0_ID, i)
                    || eq(d, e, t::ITEMPROP, "articleBody")
                    || contains_lowered(i, 'B', "articlebody")
                    || i == "articleContent"
            }
        })
    },
    |d, ctx| select_first(d, ctx, |d, e| d.tag(e) == t::ARTICLE),
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, ARTICLE_DIV_MAIN_SECTION) && {
                let (c, i) = (attr(d, e, t::CLASS), attr(d, e, t::ID));
                has(&BODY2_CLASS, c)
                    || ["postarea", "art-postcontent", "text", "cell", "story"].contains(&c)
                    || has(&BODY2_ID, i)
                    || i == "article"
                    || i == "story"
                    || fulltext(c)
                    || eq(d, e, t::ROLE, "article")
            }
        })
    },
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, ARTICLE_DIV_MAIN_SECTION) && {
                let (c, i) = (attr(d, e, t::CLASS), attr(d, e, t::ID));
                has(&BODY3_ID, i)
                    || has(&BODY3_CLASS, c)
                    || main_content(i)
                    || main_content(c)
                    || page_content(c)
                    || i == "content"
                    || c == "content"
            }
        })
    },
    |d, ctx| {
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

/// `contains(translate(@class, "FULTEX", "fultex"), "fulltext")`.
fn fulltext(s: &str) -> bool {
    let low: String = s
        .chars()
        .map(|c| match c {
            'F' | 'U' | 'L' | 'T' | 'E' | 'X' => c.to_ascii_lowercase(),
            _ => c,
        })
        .collect();
    low.contains("fulltext")
}

/// `contains(translate(s, "CM", "cm"), "main-content")`.
fn main_content(s: &str) -> bool {
    let low: String = s
        .chars()
        .map(|c| {
            if c == 'C' || c == 'M' {
                c.to_ascii_lowercase()
            } else {
                c
            }
        })
        .collect();
    low.contains("main-content")
}

/// `contains(translate(s, "CP", "cp"), "page-content")`.
fn page_content(s: &str) -> bool {
    let low: String = s
        .chars()
        .map(|c| {
            if c == 'C' || c == 'P' {
                c.to_ascii_lowercase()
            } else {
                c
            }
        })
        .collect();
    low.contains("page-content")
}

fn precedes(d: &Document, ctx: NodeId, a: NodeId, b: NodeId) -> bool {
    let mut c = d.traverse_next(ctx, ctx);
    while c != crate::html::tree::NIL {
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

litset!(COMMENTS0_ID, ["commentlist", "comment-list"]);
litset!(
    COMMENTS0_CLASS,
    [
        "commentlist",
        "comment-page",
        "comments-list",
        "comments-content",
        "post-comments"
    ]
);
litset!(COMMENTS1_ID, ["^comments", "^comment-"]);
litset!(
    COMMENTS1_CLASS,
    ["^comments", "^Comments", "^comment-", "article-comments"]
);
litset!(COMMENTS2_ID, ["^comol", "^disqus_thread", "^dsq-comments"]);

/// COMMENTS_XPATH.
pub static COMMENTS_XPATH: &[XFirst] = &[
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, DIV_LIST_SECTION)
                && (has(&COMMENTS0_ID, attr(d, e, t::ID))
                    || has(&COMMENTS0_CLASS, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, DIV_LIST_SECTION)
                && (has(&COMMENTS1_ID, attr(d, e, t::ID))
                    || has(&COMMENTS1_CLASS, attr(d, e, t::CLASS)))
        })
    },
    |d, ctx| {
        select_first(d, ctx, |d, e| {
            tag_in(d, e, DIV_LIST_SECTION) && has(&COMMENTS2_ID, attr(d, e, t::ID))
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

litset!(
    REMOVE_COMMENTS_ID,
    ["^comol", "^disqus_thread", "^dsq-comments"]
);
litset!(REMOVE_COMMENTS_CLASS, ["article-comments", "post-comments"]);

/// `starts-with(translate(s, "C", "c"), 'comment')`.
fn starts_comment(s: &str) -> bool {
    s.strip_prefix('c')
        .or_else(|| s.strip_prefix('C'))
        .is_some_and(|r| r.starts_with("omment"))
}

/// REMOVE_COMMENTS_XPATH.
pub static REMOVE_COMMENTS_XPATH: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, DIV_LIST_SECTION) && {
            let (c, i) = (attr(d, e, t::CLASS), attr(d, e, t::ID));
            starts_comment(i)
                || starts_comment(c)
                || has(&REMOVE_COMMENTS_CLASS, c)
                || has(&REMOVE_COMMENTS_ID, i)
        }
    })
}];

litset!(PAYWALL_ID, ["paywall", "premium"]);
litset!(
    PAYWALL_CLASS,
    [
        "paid-content",
        "paidcontent",
        "obfuscated",
        "blurred",
        "restricted",
        "overlay"
    ]
);

/// PAYWALL_DISCARD_XPATH.
pub static PAYWALL_DISCARD_XPATH: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, &[t::DIV, t::P])
            && (has(&PAYWALL_ID, attr(d, e, t::ID)) || has(&PAYWALL_CLASS, attr(d, e, t::CLASS)))
    })
}];

litset!(
    OD_ID,
    [
        "related",
        "viral",
        "^shar",
        "social",
        "syndication",
        "^jp-",
        "^dpsp-content",
        "newsletter",
        "cookie",
        "tags",
        "sidebar",
        "banner",
        "menu",
        "breadcrumb",
        "bread-crumb",
        "author",
        "button",
        "premium-",
        "paywall",
        "message_container",
        "bmdh",
    ]
);
litset!(
    OD_CLASS,
    [
        "viral",
        "^shar",
        "share-",
        "social",
        "sociable",
        "syndication",
        "embedded",
        "embed",
        "newsletter",
        "subnav",
        "cookie",
        "tags",
        "tag-list",
        "sidebar",
        "banner",
        "bar",
        "meta",
        "menu",
        "^nav",
        "navbar",
        "navbox",
        "^post-nav",
        "breadcrumb",
        "bread-crumb",
        "author",
        "button",
        "rating",
        "widget",
        "attachment",
        "timestamp",
        "user-info",
        "user-profile",
        "-ad-",
        "-icon",
        "article-infos",
        "outbrain",
        "taboola",
        "criteo",
        "options",
        "expand",
        "consent",
        "modal-content",
        "paid-content",
        "paidcontent",
        "obfuscated",
        "blurred",
        " ad ",
        "permission",
        "next-",
        "side-stories",
        "related-stories",
        "most-popular",
        "mol-factbox",
        "^ZendeskForm",
        "message-container",
        "yin",
        "zlylin",
        "xg1",
        "slide",
        "viewport",
    ]
);
litset!(
    OD1_CLASS,
    [
        "comments-title",
        "nocomments",
        "^reply-",
        "-reply-",
        "message",
        "akismet",
        "suggest-links",
        "^hide-",
        "-hide-",
        "hide-print",
        " hidden",
        " hide",
        "noprint",
        "notloaded",
    ]
);
litset!(OD1_ID, ["^reply-", "reader-comments", "akismet", "hidden"]);

/// OVERALL_DISCARD_XPATH.
pub static OVERALL_DISCARD_XPATH: XList = &[
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DISCARD_SCOPE) && {
                let (c, i) = (attr(d, e, t::CLASS), attr(d, e, t::ID));
                contains_lowered(i, 'F', "footer")
                    || contains_lowered(c, 'F', "footer")
                    || contains_lowered(c, 'R', "related")
                    || contains_lowered(i, 'S', "share")
                    || contains_lowered(i, 'N', "nav")
                    || contains_lowered(attr(d, e, t::ROLE), 'N', "nav")
                    || contains_lowered(c, 'N', "navigation")
                    || contains_lowered(c, 'B', "byline")
                    || contains_lowered(c, 'I', "infoline")
                    || has(&OD_ID, i)
                    || has(&OD_CLASS, c)
                    || d.get(e, "data-component")
                        .unwrap_or("")
                        .contains("MostPopularStories")
                    || d.get(e, "data-lp-replacement-content").is_some()
            }
        })
    },
    |d, ctx| {
        select(d, ctx, |d, e| {
            let (c, i, st) = (
                attr(d, e, t::CLASS),
                attr(d, e, t::ID),
                attr(d, e, t::STYLE),
            );
            has(&OD1_CLASS, c)
                || has(&OD1_ID, i)
                || st.contains("hidden")
                || st.contains("display:none")
                || st.contains("display: none")
                || eq(d, e, t::ARIA_HIDDEN, "true")
        })
    },
];

/// TEASER_DISCARD_XPATH.
pub static TEASER_DISCARD_XPATH: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, DISCARD_SCOPE)
            && (contains_lowered(attr(d, e, t::ID), 'T', "teaser")
                || contains_lowered(attr(d, e, t::CLASS), 'T', "teaser"))
    })
}];

/// PRECISION_DISCARD_XPATH.
pub static PRECISION_DISCARD_XPATH: XList = &[
    |d, ctx| select(d, ctx, |d, e| d.tag(e) == t::HEADER),
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DISCARD_SCOPE) && {
                let (c, i) = (attr(d, e, t::CLASS), attr(d, e, t::ID));
                i.contains("bottom")
                    || c.contains("bottom")
                    || i.contains("link")
                    || c.contains("link")
                    || attr(d, e, t::STYLE).contains("border")
            }
        })
    },
];

/// DISCARD_IMAGE_ELEMENTS.
pub static DISCARD_IMAGE_ELEMENTS: XList = &[|d, ctx| {
    select(d, ctx, |d, e| {
        tag_in(d, e, DISCARD_SCOPE)
            && (attr(d, e, t::ID).contains("caption") || attr(d, e, t::CLASS).contains("caption"))
    })
}];

litset!(
    CD_CLASS,
    [
        "comments-title",
        "nocomments",
        "^reply-",
        "-reply-",
        "message",
        "signin",
        "akismet"
    ]
);
litset!(CD_ID, ["^reply-", "akismet"]);

/// COMMENTS_DISCARD_XPATH.
pub static COMMENTS_DISCARD_XPATH: XList = &[
    |d, ctx| {
        select(d, ctx, |d, e| {
            tag_in(d, e, DIV_SECTION) && attr(d, e, t::ID).starts_with("respond")
        })
    },
    |d, ctx| select(d, ctx, |d, e| tag_in(d, e, &[t::CITE, t::QUOTE])),
    |d, ctx| {
        select(d, ctx, |d, e| {
            has(&CD_CLASS, attr(d, e, t::CLASS))
                || has(&CD_ID, attr(d, e, t::ID))
                || attr(d, e, t::STYLE).contains("display:none")
        })
    },
];

/// The RAW_TREE_PRUNE / REMOVE_COMMENTS_AND_LISTS slots of the profile, which 1.11.0 does not
/// have: nothing is selected.
pub static NONE: XList = &[|_, _| Vec::new()];

/// settings.BASIC_CLEAN_XPATH: `.//aside|.//footer|.//script|.//style`.
pub fn basic_clean(d: &Document, ctx: NodeId) -> Vec<NodeId> {
    select(d, ctx, |d, e| {
        tag_in(d, e, &[t::ASIDE, t::FOOTER, t::SCRIPT, t::STYLE])
    })
}

/// external.SANITIZED_XPATH is non-empty (1.11.0's list, without fencedframe).
pub fn sanitized_any(d: &Document, body: NodeId) -> bool {
    const TAGS: &[Name] = &[
        t::ASIDE,
        t::AUDIO,
        t::BUTTON,
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
    select_first(d, body, |d, e| tag_in(d, e, TAGS)).is_some()
}

/// recover_wild_text's search: `.//blockquote|.//code|.//p|.//pre|.//q|.//quote|.//table|
/// .//div[contains(@class, 'w3-code')]`, plus `|.//div|.//lb|.//list` in recall mode.
pub fn wild_text_search(d: &Document, ctx: NodeId, recall: bool) -> Vec<NodeId> {
    select(d, ctx, |d, e| {
        let g = d.tag(e);
        [
            t::BLOCKQUOTE,
            t::CODE,
            t::P,
            t::PRE,
            t::Q,
            t::QUOTE,
            t::TABLE,
        ]
        .contains(&g)
            || (g == t::DIV && (recall || attr(d, e, t::CLASS).contains("w3-code")))
            || (recall && (g == t::LB || g == t::LIST))
    })
}
