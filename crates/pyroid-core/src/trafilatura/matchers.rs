// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! The regular expressions Trafilatura, its readability fork and jusText apply, written out as
//! matchers with Python `re` semantics (`\s`, `\w`, `$`, `re.IGNORECASE`) for the caller's
//! Unicode version.
//!
//! Most patterns are finite alternations: they are expanded into literal sets ([`LitSet`]),
//! with `^`-anchored literals checked at the start only. Case-insensitive sets match on the
//! text folded by [`pystr::icase_fold`] (ASCII letters plus the four non-ASCII characters
//! Python's `re.IGNORECASE` equates with `i`, `k` and `s`).

use std::sync::LazyLock;

use super::pystr::{self, icase_fold, is_space, is_word};
use crate::ftfy::UnicodeVersion;

/// A set of literals, some anchored at the start (`^` prefix in the source list).
#[derive(Debug)]
pub struct LitSet {
    anchored: Vec<&'static [u8]>,
    by_first: Vec<Vec<&'static [u8]>>,
    first: [bool; 256],
}

impl LitSet {
    pub fn new(literals: &[&'static str]) -> Self {
        let mut s = LitSet {
            anchored: Vec::new(),
            by_first: vec![Vec::new(); 256],
            first: [false; 256],
        };
        for l in literals {
            if let Some(a) = l.strip_prefix('^') {
                s.anchored.push(a.as_bytes());
            } else if let Some(&b) = l.as_bytes().first() {
                s.by_first[b as usize].push(l.as_bytes());
                s.first[b as usize] = true;
            }
        }
        s
    }

    /// `re.search(pattern, hay)` for the alternation of the literals.
    pub fn search(&self, hay: &str) -> bool {
        let h = hay.as_bytes();
        if self.anchored.iter().any(|a| h.starts_with(a)) {
            return true;
        }
        for i in 0..h.len() {
            if self.first[h[i] as usize]
                && self.by_first[h[i] as usize]
                    .iter()
                    .any(|n| h[i..].starts_with(n))
            {
                return true;
            }
        }
        false
    }

    /// Case-insensitive search (`re.IGNORECASE`); the literals must be lower-case ASCII.
    pub fn search_icase(&self, hay: &str) -> bool {
        if hay.is_ascii() {
            if !hay.bytes().any(|b| b.is_ascii_uppercase()) {
                return self.search(hay);
            }
            return self.search(&hay.to_ascii_lowercase());
        }
        let folded: String = hay.chars().map(icase_fold).collect();
        self.search(&folded)
    }
}

macro_rules! litset {
    ($name:ident, [$($l:literal),* $(,)?]) => {
        pub static $name: LazyLock<LitSet> = LazyLock::new(|| LitSet::new(&[$($l),*]));
    };
}

// ---- xpaths.py ---------------------------------------------------------------------------------

// RAW_TREE_PRUNE_XPATH
litset!(
    MVP_OR_SCROLL,
    [
        "mvp-post-add-box",
        "mvp-post-add-wrap",
        "infinite-scroll",
        "infinitescroll"
    ]
);
litset!(MVP, ["mvp-post-add-box", "mvp-post-add-wrap"]);

// BODY_XPATH
litset!(
    BODY0_ID,
    [
        "entry-content",
        "article-content",
        "art-content",
        "article__content",
        "article-body",
        "article__body",
        "articlebody",
        "articleBody",
        "body-text",
    ]
);
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
        "page-content",
        "text-content",
        "article-content",
        "art-content",
        "article__content",
        "article-body",
        "article__body",
        "articlebody",
        "articleBody",
        "ArticleContent",
        "body-text",
        "article__container",
    ]
);
litset!(BODY2_ID, ["^primary", "story-body"]);
litset!(BODY2_FULLTEXT, ["fulltext"]);
litset!(
    BODY2_CLASS,
    [
        "^article ",
        "post-bodycopy",
        "story-content",
        "storycontent",
        "theme-content",
        "blog-content",
        "section-content",
        "single-content",
        "single-post",
        "main-column",
        "wpb_text_column",
        "story-body",
        "field-body",
    ]
);
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

// COMMENTS_XPATH
litset!(COMMENTS0_IDCLASS, ["comment-list", "commentlist"]);
litset!(
    COMMENTS0_CLASS,
    ["comment-page", "comments-content", "post-comments"]
);
litset!(COMMENTS1_IDCLASS, ["^comments", "^comment-"]);
litset!(COMMENTS1_CLASS, ["^Comments", "article-comments"]);
litset!(COMMENTS2_ID, ["^comol", "^disqus_thread", "^dsq-comments"]);

// COMMENTS_DISCARD_XPATH
litset!(
    COMMENTS_DISCARD_CLASS,
    [
        "comments-title",
        "nocomments",
        "-reply-",
        "message",
        "signin"
    ]
);
litset!(COMMENTS_DISCARD_IDCLASS, ["^reply-", "akismet"]);

// OVERALL_DISCARD_XPATH
litset!(
    OD_ID,
    [
        "^shar",
        "social",
        "viral",
        "newsletter",
        "syndication",
        "tags",
        "sidebar",
        "banner",
        "bread-crumb",
        "breadcrumb",
        "button",
        "author",
        "^jp-",
        "^dpsp-content",
        "bmdh",
        "footer",
        "Footer",
        "share",
        "Share",
        "nav",
        "Nav",
        "menu",
        "related",
        "message-container",
        "premium",
        "^ad-",
    ]
);
litset!(
    OD_CLASS,
    [
        "^shar",
        "social",
        "viral",
        "newsletter",
        "syndication",
        "tags",
        "sidebar",
        "banner",
        "bread-crumb",
        "breadcrumb",
        "button",
        "author",
        "^nav",
        "^post-nav",
        "^ZendeskForm",
        "subnav",
        "avigation",
        "navbar",
        "navbox",
        "menu",
        "bar",
        " ad ",
        "-ad-",
        "^ad-",
        " ad-",
        "dfpAd",
        "dfpad",
        "outbrain",
        "taboola",
        "criteo",
        "paid-content",
        "paidcontent",
        "widget",
        "footer",
        "Footer",
        "byline",
        "Byline",
        "share-",
        "sociable",
        "embedded",
        "embed",
        "tag-list",
        "consent",
        "modal-content",
        "permission",
        "elated",
        "next-",
        "-stories",
        "most-popular",
        "more-on",
        "meta",
        "rating",
        "attachment",
        "timestamp",
        "user-info",
        "user-profile",
        "-icon",
        "article-infos",
        "message-container",
        "slide",
        "viewport",
        "overlay",
        "options",
        "expand",
        "obfuscated",
        "blurred",
        "mol-factbox",
        "yin",
        "zlylin",
        "nfoline",
    ]
);
litset!(OD1_ID, ["reader-comments", "akismet"]);
litset!(
    OD1_CLASS,
    [
        "^hide-",
        "comments-title",
        "nocomments",
        "-reply-",
        "message",
        "akismet",
        "suggest-links",
        "-hide-",
        "hide-print",
        " hidden",
        " hide",
        "noprint",
        "notloaded",
    ]
);

// settings._COOKIE_CONSENT_RE (BASIC_CLEAN_XPATH, with the 'i' flag)
litset!(
    COOKIE_CONSENT,
    [
        "cookiebanner",
        "cookie-banner",
        "cookie_banner",
        "cookiebar",
        "cookie-bar",
        "cookie_bar",
        "cookieconsent",
        "cookie-consent",
        "cookie_consent",
        "cookielaw",
        "cookie-law",
        "cookie_law",
        "cookienotice",
        "cookie-notice",
        "cookie_notice",
        "cookiepolicy",
        "cookie-policy",
        "cookie_policy",
        "cookiedescription",
        "cookie-description",
        "cookie_description",
        "noticecookie",
        "notice-cookie",
        "notice_cookie",
        "notice--cookie",
        "notice-_cookie",
        "notice_-cookie",
        "notice__cookie",
        "consentbanner",
        "consent-banner",
        "consent_banner",
        "consentmanager",
        "consent-manager",
        "consent_manager",
        "consentsdk",
        "consent-sdk",
        "consent_sdk",
        "borlabs",
        "cookiebot",
        "cmplz",
        "onetrust",
        "moovegdpr",
        "moove-gdpr",
        "moove_gdpr",
    ]
);

/// `^(?:[Cc]omment(?!ary)|...)`: starts with "comment"/"Comment" not followed by "ary".
fn starts_comment_not_ary(s: &str) -> bool {
    (s.starts_with("comment") || s.starts_with("Comment")) && !s[7..].starts_with("ary")
}
/// `^(?:[Cc]omment(?!ary)|comol|disqus_thread|dsq-comments)`
pub fn comments_id(s: &str) -> bool {
    starts_comment_not_ary(s)
        || s.starts_with("comol")
        || s.starts_with("disqus_thread")
        || s.starts_with("dsq-comments")
}
/// `^(?:[Cc]omment(?!ary)|comol)`
pub fn comments_id_short(s: &str) -> bool {
    starts_comment_not_ary(s) || s.starts_with("comol")
}
/// `^[Cc]omment(?!ary)|(?:article|post)-comments`
pub fn comments_class(s: &str) -> bool {
    starts_comment_not_ary(s) || s.contains("article-comments") || s.contains("post-comments")
}

/// `(^|\s)WORD|WORD(\s|$)` (`both`: `(^|\s)WORD(\s|$)`), Python `\s` and `$`.
fn word_at_boundary(s: &str, word: &str, both: bool) -> bool {
    let mut from = 0;
    while let Some(p) = s[from..].find(word) {
        let i = from + p;
        let end = i + word.len();
        let before = i == 0 || s[..i].chars().next_back().is_some_and(is_space);
        let after = end == s.len() || s[end..].chars().next().is_some_and(is_space);
        if (both && before && after) || (!both && (before || after)) {
            return true;
        }
        from = i + 1;
    }
    false
}
/// PRECISION_DISCARD_XPATH `(^|\s)bottom|bottom(\s|$)`
pub fn bottom(s: &str) -> bool {
    word_at_boundary(s, "bottom", false)
}
/// PRECISION_DISCARD_XPATH `(^|\s)link(\s|$)`
pub fn link_word(s: &str) -> bool {
    word_at_boundary(s, "link", true)
}

/// core._DISCUSSION_FORUM_POSTING_RE.search:
/// `"@type"\s*:\s*"DiscussionForumPosting"|"@type"\s*:\s*\[[^\]]*"DiscussionForumPosting"`.
pub fn forum_posting(s: &str) -> bool {
    const KEY: &str = "\"@type\"";
    const VAL: &str = "\"DiscussionForumPosting\"";
    let mut from = 0;
    while let Some(p) = s[from..].find(KEY) {
        let i = from + p + KEY.len();
        let rest = pystr::lstrip(&s[i..]);
        if let Some(r) = rest.strip_prefix(':') {
            let r = pystr::lstrip(r);
            if r.starts_with(VAL) {
                return true;
            }
            if let Some(list) = r.strip_prefix('[') {
                let span = &list[..list.find(']').unwrap_or(list.len())];
                if span.contains(VAL) {
                    return true;
                }
            }
        }
        from = from + p + 1;
    }
    false
}

// ---- utils.py ----------------------------------------------------------------------------------

/// utils.RE_FILTER.match on one line (no line breaks): `\W*(Drucken|E-?Mail|Facebook|...|
/// Mehr zum Thema:?|More on this.{,8}$)$` with re.IGNORECASE.
pub fn re_filter_match(line: &str, v: UnicodeVersion) -> bool {
    const EXACT: &[&str] = &[
        "drucken",
        "email",
        "e-mail",
        "facebook",
        "flipboard",
        "google",
        "instagram",
        "linkedin",
        "mail",
        "pdf",
        "pinterest",
        "pocket",
        "print",
        "qq",
        "reddit",
        "twitter",
        "wechat",
        "weibo",
        "whatsapp",
        "xing",
        "mehr zum thema",
        "mehr zum thema:",
    ];
    const MORE: &str = "more on this";
    // \W* then a word character starts every alternative, so the match starts after the
    // leading non-word characters
    let start = line
        .char_indices()
        .find(|&(_, c)| is_word(c, v))
        .map_or(line.len(), |(i, _)| i);
    let rest: Vec<char> = line[start..].chars().map(icase_fold).collect();
    let eq = |lit: &str| rest.iter().copied().eq(lit.chars());
    if EXACT.iter().any(|l| eq(l)) {
        return true;
    }
    // More on this.{,8}$ (then $): up to 8 characters other than "\n" after "more on this";
    // a final "\n" may also follow, but a line holds none
    let n = MORE.chars().count();
    rest.len() >= n
        && rest.len() <= n + 8
        && rest[..n].iter().copied().eq(MORE.chars())
        && !rest[n..].contains(&'\n')
}

/// utils.IMAGE_EXTENSION.search: `[^\s]+\.(avif|bmp|gif|hei[cf]|jpe?g|png|webp)(\b|$)`, re.I.
pub fn image_extension(s: &str, v: UnicodeVersion) -> bool {
    const EXT: &[&str] = &[
        "avif", "bmp", "gif", "heic", "heif", "jpeg", "jpg", "png", "webp",
    ];
    let cs: Vec<char> = s.chars().collect();
    for (i, &c) in cs.iter().enumerate() {
        if c != '.' || i == 0 || is_space(cs[i - 1]) {
            continue;
        }
        for e in EXT {
            let k = e.chars().count();
            if i + 1 + k > cs.len() {
                continue;
            }
            if !cs[i + 1..i + 1 + k]
                .iter()
                .map(|&x| icase_fold(x))
                .eq(e.chars())
            {
                continue;
            }
            // \b after a word character: end of string or a non-word character next
            if cs.get(i + 1 + k).is_none_or(|&n| !is_word(n, v)) {
                return true;
            }
        }
    }
    false
}

// ---- readability_lxml.py -----------------------------------------------------------------------

litset!(
    UNLIKELY,
    [
        "combx",
        "comment",
        "community",
        "disqus",
        "extra",
        "foot",
        "header",
        "menu",
        "remark",
        "rss",
        "shoutbox",
        "sidebar",
        "sponsor",
        "ad-break",
        "agegate",
        "pagination",
        "pager",
        "popup",
        "tweet",
        "twitter",
    ]
);
litset!(
    OK_MAYBE,
    [
        "and", "article", "body", "column", "content", "main", "shadow"
    ]
);
litset!(
    POSITIVE,
    [
        "article",
        "body",
        "content",
        "entry",
        "hentry",
        "main",
        "page",
        "pagination",
        "post",
        "text",
        "blog",
        "story",
    ]
);
litset!(
    NEGATIVE,
    [
        "button", "combx", "comment", "com-", "contact", "figure", "foot", "footer", "footnote",
        "form", "input", "masthead", "media", "meta", "outbrain", "promo", "related", "scroll",
        "shoutbox", "sidebar", "sponsor", "shopping", "tags", "tool", "widget",
    ]
);
litset!(
    VIDEO,
    [
        "http://youtube.com",
        "http://www.youtube.com",
        "https://youtube.com",
        "https://www.youtube.com",
        "http://vimeo.com",
        "http://www.vimeo.com",
        "https://vimeo.com",
        "https://www.vimeo.com",
    ]
);

/// readability DOT_SPACE.search: `\.( |$)` (`$`: the end, or before a final "\n").
pub fn dot_space(s: &str) -> bool {
    let b = s.as_bytes();
    b.iter().enumerate().any(|(i, &x)| {
        x == b'.'
            && (b.get(i + 1).is_none_or(|&n| n == b' ')
                || (b.get(i + 1) == Some(&b'\n') && i + 2 == b.len()))
    })
}

// ---- baseline.py -------------------------------------------------------------------------------

// baseline._JSON_HOOKS_RE
litset!(
    JSON_HOOKS,
    [
        "articleBody",
        "reviewBody",
        "recipeInstructions",
        "acceptedAnswer",
        "\"Product\"",
        "\"VideoObject\"",
        "\"HowTo\"",
    ]
);

/// baseline._HTML_TAG_NAMES, longest first is not needed: the pattern only has to match somewhere.
const HTML_TAG_NAMES: &[&str] = &[
    "a",
    "abbr",
    "address",
    "article",
    "aside",
    "b",
    "blockquote",
    "body",
    "br",
    "caption",
    "cite",
    "code",
    "dd",
    "del",
    "div",
    "dl",
    "dt",
    "em",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "i",
    "img",
    "ins",
    "kbd",
    "li",
    "main",
    "mark",
    "nav",
    "ol",
    "p",
    "pre",
    "q",
    "quote",
    "s",
    "section",
    "small",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "time",
    "title",
    "tr",
    "u",
    "ul",
];

/// baseline._HTML_MARKUP.search, re.IGNORECASE:
/// `</(NAMES)>|<(NAMES)(\s[^<>]*=[^<>]*)?/?>`.
pub fn html_markup(s: &str) -> bool {
    let cs: Vec<char> = s.chars().map(icase_fold).collect();
    let starts = |at: usize, w: &str| -> bool {
        let k = w.len();
        at + k <= cs.len() && cs[at..at + k].iter().copied().eq(w.chars())
    };
    for i in 0..cs.len() {
        if cs[i] != '<' {
            continue;
        }
        if cs.get(i + 1) == Some(&'/') {
            if HTML_TAG_NAMES
                .iter()
                .any(|n| starts(i + 2, n) && cs.get(i + 2 + n.len()) == Some(&'>'))
            {
                return true;
            }
            continue;
        }
        for n in HTML_TAG_NAMES {
            if !starts(i + 1, n) {
                continue;
            }
            let j = i + 1 + n.len();
            // "/?>" right away
            if cs.get(j) == Some(&'>') || (cs.get(j) == Some(&'/') && cs.get(j + 1) == Some(&'>')) {
                return true;
            }
            // \s[^<>]*=[^<>]*/?> : the first '<' or '>' after the space must be '>', with an '='
            // before it ([^<>]* takes the optional '/')
            if cs.get(j).is_some_and(|&c| is_space(c)) {
                let mut k = j + 1;
                let mut eq = false;
                while k < cs.len() && cs[k] != '<' && cs[k] != '>' {
                    eq |= cs[k] == '=';
                    k += 1;
                }
                if eq && cs.get(k) == Some(&'>') {
                    return true;
                }
            }
        }
    }
    false
}

// ---- justext -----------------------------------------------------------------------------------

/// justext.utils.normalize_whitespace: every run of `\s` becomes "\n" if it holds "\n" or "\r",
/// else " ".
pub fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut run: Option<bool> = None;
    for c in s.chars() {
        if is_space(c) {
            let nl = c == '\n' || c == '\r';
            run = Some(run.unwrap_or(false) || nl);
        } else {
            if let Some(nl) = run.take() {
                out.push(if nl { '\n' } else { ' ' });
            }
            out.push(c);
        }
    }
    if let Some(nl) = run {
        out.push(if nl { '\n' } else { ' ' });
    }
    out
}

// ---- xml.py ------------------------------------------------------------------------------------

/// xml._UNESCAPED_BRACKET_RE.sub(lambda m: m[1] or f"\\{m[3]}", text), DOTALL:
/// `((?<![\\`])(`+)(?!`).+?(?<!`)\2(?!`))|(?<!\\)([\[\]])` -- code spans are kept, other
/// unescaped brackets get a backslash.
pub fn escape_brackets(text: &str) -> String {
    let cs: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 4);
    let mut i = 0;
    while i < cs.len() {
        // alternative 1: a code span starting here
        if cs[i] == '`' && (i == 0 || (cs[i - 1] != '\\' && cs[i - 1] != '`')) {
            // (`+)(?!`): the whole run of backticks (backtracking to a shorter run would leave a
            // backtick next, failing the lookahead)
            let mut r = i;
            while r < cs.len() && cs[r] == '`' {
                r += 1;
            }
            let n = r - i;
            // .+? then (?<!`)\2(?!`): the first closing run of exactly n backticks, after at
            // least one character, not preceded by '`' and not followed by '`'
            let mut found = None;
            let mut k = r + 1;
            while k + n <= cs.len() {
                if cs[k..k + n].iter().all(|&c| c == '`')
                    && cs[k - 1] != '`'
                    && cs.get(k + n) != Some(&'`')
                {
                    found = Some(k + n);
                    break;
                }
                k += 1;
            }
            if let Some(end) = found {
                out.extend(&cs[i..end]);
                i = end;
                continue;
            }
        }
        let c = cs[i];
        if (c == '[' || c == ']') && (i == 0 || cs[i - 1] != '\\') {
            out.push('\\');
        }
        out.push(c);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    const V: UnicodeVersion = UnicodeVersion::V15_1;

    #[test]
    fn filter() {
        assert!(re_filter_match("Facebook", V));
        assert!(re_filter_match("  * PRİNT", V));
        assert!(re_filter_match("More on this topic", V));
        assert!(re_filter_match("E-Mail", V));
        assert!(re_filter_match("Mehr zum Thema:", V));
        assert!(!re_filter_match("More on this topic and others", V));
        assert!(!re_filter_match("Facebook page", V));
        assert!(!re_filter_match("", V));
    }

    #[test]
    fn images() {
        assert!(image_extension("a/b.JPG", V));
        assert!(image_extension("x.jpeg?w=1", V));
        assert!(!image_extension(".png", V));
        assert!(!image_extension("a .png", V));
        assert!(!image_extension("a.pngx", V));
        assert!(image_extension("a.png_x.gif", V));
        assert!(image_extension("a.heiC-", V));
    }

    #[test]
    fn markup() {
        assert!(html_markup("a <p>b"));
        assert!(html_markup("</P>"));
        assert!(html_markup("<br/>"));
        assert!(html_markup("<div class=x>"));
        assert!(!html_markup("i<b and c>d"));
        assert!(!html_markup("<div class>"));
        assert!(html_markup("<ſtrong>"));
    }

    #[test]
    fn brackets() {
        assert_eq!(escape_brackets("a[b]"), "a\\[b\\]");
        assert_eq!(escape_brackets("a\\[b"), "a\\[b");
        assert_eq!(escape_brackets("`a[b]` [c]"), "`a[b]` \\[c\\]");
        assert_eq!(escape_brackets("``x`` ["), "``x`` \\[");
        assert_eq!(escape_brackets("`unclosed ["), "`unclosed \\[");
    }

    #[test]
    fn boundaries() {
        assert!(bottom("page-bottom"));
        assert!(bottom("bottom-bar"));
        assert!(!bottom("Padding-bottom-lg"));
        assert!(link_word("a link b"));
        assert!(!link_word("permalink"));
        assert!(forum_posting(
            r#"{"@type" : [ "X", "DiscussionForumPosting"]}"#
        ));
        assert!(!forum_posting(
            r#"{"@type": ["X"], "a": "DiscussionForumPosting"}"#
        ));
        assert!(dot_space("a. b"));
        assert!(dot_space("a.\n"));
        assert!(!dot_space("a.b"));
    }

    #[test]
    fn sets() {
        assert!(OD_CLASS.search("x ad-y"));
        assert!(OD_CLASS.search("ad-y"));
        assert!(!OD_CLASS.search("xad-y"));
        assert!(COOKIE_CONSENT.search_icase("Cookie-Banner"));
        assert!(VIDEO.search_icase("HTTPS://www.YouTube.com/embed"));
    }
}
