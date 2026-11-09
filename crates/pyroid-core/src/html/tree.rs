// Derived from lxml 6.1.3 (BSD-3-Clause), © Infrae; modified.
//! The document tree: an arena with lxml's element model, written by [`super::parser`] and read
//! and edited by the extraction code without a conversion.
//!
//! libxml2 keeps text in sibling text nodes; lxml exposes them as `.text` (the text before an
//! element's first child) and `.tail` (the text after an element, up to its next sibling). The
//! arena stores that model directly:
//!
//! - every node owns its `text` and `tail`. `None` is "no text node", `Some("")` an empty one:
//!   lxml returns `None` and `''` for them, and Trafilatura tells the two apart
//!   (`e.text is None`, `" ".join(itertext())`);
//! - siblings are linked like libxml2's element-like nodes (elements, comments, processing
//!   instructions), so lxml's tree walkers can be replayed step by step. Removing or appending
//!   an element moves its tail with it (lxml `remove`/`append`); [`Document::strip_tags`]
//!   splices text and children into the parent (`_replaceNodeByChildren`);
//! - [`Iter`] reproduces lxml's `ElementDepthFirstIterator`, which computes the next match when
//!   it returns the current one, so a loop body that edits the tree changes what follows exactly
//!   as in lxml;
//! - element and attribute names are interned [`Name`]s: a static table with fixed ids (the
//!   constants in [`names`]) plus whatever names a document brings;
//! - nodes are never freed: a detached subtree stays in the arena, like an lxml element kept
//!   alive by a Python reference. [`Document::top`] stands for lxml's `getroottree().getroot()`;
//! - walks, copies, dumps and drops are iterative. The parser limits depth to 256, but edits
//!   can nest deeper, and nothing here recurses.

use std::fmt::Write as _;
use std::sync::LazyLock;

use rustc_hash::FxHashMap;

/// Index of a node in [`Document::nodes`].
pub type NodeId = u32;

/// "No node" in the link fields of [`Node`].
pub const NIL: NodeId = u32::MAX;

/// An interned element or attribute name. Ids below [`names::STATIC_COUNT`] are the same in
/// every document (see [`names`]); the others belong to one [`Document`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Name(pub u32);

impl Name {
    /// Whether the name is one of the static [`names`] (the same id in every document).
    #[inline]
    pub const fn is_static(self) -> bool {
        self.0 < names::STATIC_COUNT
    }
}

macro_rules! static_names {
    ($($c:ident $s:literal)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms, dead_code)]
        #[repr(u32)]
        enum Known { $($c,)* Count }

        /// Names with fixed ids: the elements libxml2's HTML parser knows, other common HTML
        /// and SVG elements, the TEI-like elements Trafilatura's output uses, and common
        /// attribute names. The parser interns every other name per document.
        pub mod names {
            use super::{Known, Name};

            $(#[doc = concat!("`", $s, "`")] pub const $c: Name = Name(Known::$c as u32);)*

            /// Number of static names.
            pub const STATIC_COUNT: u32 = Known::Count as u32;

            /// The static names, indexed by id.
            pub(crate) static STRINGS: [&str; STATIC_COUNT as usize] = [$($s),*];
        }
    };
}

static_names! {
    NONE ""
    // Elements (ids below 256, so a `TagSet` of elements stays in its bit array).
    A "a" ABBR "abbr" ACRONYM "acronym" ADDRESS "address" APPLET "applet" AREA "area"
    ARTICLE "article" ASIDE "aside" AUDIO "audio" B "b" BASE "base" BASEFONT "basefont" BDI "bdi"
    BDO "bdo" BIG "big" BLINK "blink" BLOCKQUOTE "blockquote" BODY "body" BR "br"
    BUTTON "button" CANVAS "canvas" CAPTION "caption" CENTER "center" CITE "cite" CODE "code"
    COL "col" COLGROUP "colgroup" DATA "data" DATALIST "datalist" DD "dd" DEL "del"
    DETAILS "details" DFN "dfn" DIALOG "dialog" DIR "dir" DIV "div" DL "dl" DT "dt" EM "em"
    EMBED "embed" FENCEDFRAME "fencedframe" FIELDSET "fieldset" FIGCAPTION "figcaption"
    FIGURE "figure" FONT "font" FOOTER "footer" FORM "form" FRAME "frame" FRAMESET "frameset"
    H1 "h1" H2 "h2" H3 "h3" H4 "h4" H5 "h5" H6 "h6" HEAD "head" HEADER "header" HGROUP "hgroup"
    HR "hr" HTML "html" I "i" IFRAME "iframe" IMG "img" INPUT "input" INS "ins"
    ISINDEX "isindex" KBD "kbd" LABEL "label" LEGEND "legend" LI "li" LINK "link"
    LISTING "listing" MAIN "main" MAP "map" MARK "mark" MARQUEE "marquee" MATH "math"
    MENU "menu" MENUITEM "menuitem" META "meta" METER "meter" NAV "nav" NOBR "nobr"
    NOEMBED "noembed" NOFRAMES "noframes" NOINDEX "noindex" NOSCRIPT "noscript"
    OBJECT "object" OL "ol" OPTGROUP "optgroup" OPTION "option" OUTPUT "output" P "p"
    PARAM "param" PICTURE "picture" PLAINTEXT "plaintext" PRE "pre" PROGRESS "progress" Q "q"
    RP "rp" RT "rt" RTC "rtc" RUBY "ruby" S "s" SAMP "samp" SCRIPT "script" SEARCH "search"
    SECTION "section" SELECT "select" SLOT "slot" SMALL "small" SOURCE "source" SPAN "span"
    STRIKE "strike" STRONG "strong" STYLE "style" SUB "sub" SUMMARY "summary" SUP "sup"
    SVG "svg" TABLE "table" TBODY "tbody" TD "td" TEMPLATE "template" TEXTAREA "textarea"
    TFOOT "tfoot" TH "th" THEAD "thead" TIME "time" TITLE "title" TR "tr" TRACK "track"
    TT "tt" U "u" UL "ul" VAR "var" VIDEO "video" WBR "wbr" XMP "xmp" O_P "o:p"
    ANNOTATION "annotation" CIRCLE "circle" CLIPPATH "clippath" DEFS "defs" ELLIPSE "ellipse"
    FILTER "filter" FOREIGNOBJECT "foreignobject" G "g" IMAGE "image" LINE "line"
    LINEARGRADIENT "lineargradient" MASK "mask" PATH "path" POLYGON "polygon"
    POLYLINE "polyline" RADIALGRADIENT "radialgradient" RECT "rect" STOP "stop"
    SYMBOL "symbol" TEXT "text" TSPAN "tspan" USE "use"
    // Trafilatura's output elements.
    AB "ab" CELL "cell" DONE "done" GRAPHIC "graphic" H "h" HI "hi" ITEM "item" LB "lb"
    LIST "list" QUOTE "quote" REF "ref" ROW "row"
    // Attributes (names shared with an element, such as `style`, use the element's id).
    ID "id" CLASS "class" HREF "href" SRC "src" ALT "alt" REL "rel" TYPE "type" NAME "name"
    CONTENT "content" LANG "lang" XML_LANG "xml:lang" WIDTH "width" HEIGHT "height"
    TARGET "target" VALUE "value" PROPERTY "property" ROLE "role" SRCSET "srcset"
    SIZES "sizes" LOADING "loading" DECODING "decoding" ONCLICK "onclick" ONLOAD "onload"
    ONERROR "onerror" ONSUBMIT "onsubmit" ONCHANGE "onchange" ONMOUSEOVER "onmouseover"
    ONMOUSEOUT "onmouseout" ACTION "action" METHOD "method" FOR "for"
    PLACEHOLDER "placeholder" CHARSET "charset" HTTP_EQUIV "http-equiv" ITEMPROP "itemprop"
    ITEMSCOPE "itemscope" ITEMTYPE "itemtype" ITEMID "itemid" TABINDEX "tabindex"
    ARIA_LABEL "aria-label" ARIA_HIDDEN "aria-hidden" ARIA_EXPANDED "aria-expanded"
    ARIA_LABELLEDBY "aria-labelledby" ARIA_CONTROLS "aria-controls"
    ARIA_HASPOPUP "aria-haspopup" ARIA_CURRENT "aria-current"
    ARIA_DESCRIBEDBY "aria-describedby" DATA_SRC "data-src" DATA_ID "data-id" XMLNS "xmlns"
    XMLNS_XLINK "xmlns:xlink" XLINK_HREF "xlink:href" VIEWBOX "viewbox" FILL "fill" D "d"
    STROKE "stroke" BORDER "border" ALIGN "align" VALIGN "valign" COLSPAN "colspan"
    ROWSPAN "rowspan" CELLPADDING "cellpadding" CELLSPACING "cellspacing" BGCOLOR "bgcolor"
    COLOR "color" FACE "face" SIZE "size" MEDIA "media" ASYNC "async"
    CROSSORIGIN "crossorigin" INTEGRITY "integrity" REFERRERPOLICY "referrerpolicy"
    FRAMEBORDER "frameborder" ALLOWFULLSCREEN "allowfullscreen" ALLOW "allow"
    SCROLLING "scrolling" DATETIME "datetime" AUTOCOMPLETE "autocomplete" ENCTYPE "enctype"
    NONCE "nonce" MAXLENGTH "maxlength" AUTOPLAY "autoplay" CONTROLS "controls" LOOP "loop"
    MUTED "muted" POSTER "poster" PRELOAD "preload" HREFLANG "hreflang" AS "as"
    SCOPE "scope" HEADERS "headers" HIDDEN "hidden" REQUIRED "required" REND "rend"
    // HTMLtree.c htmlBooleanAttrs.
    CHECKED "checked" COMPACT "compact" DECLARE "declare" DEFER "defer" DISABLED "disabled"
    ISMAP "ismap" MULTIPLE "multiple" NOHREF "nohref" NORESIZE "noresize" NOSHADE "noshade"
    NOWRAP "nowrap" READONLY "readonly" SELECTED "selected"
}

// Static names fit `TagSet`'s bit array, and element names (the last is `row`) stay below 256.
const _: () = assert!(names::STATIC_COUNT <= 512 && names::ROW.0 < 256);

static STATIC_LOOKUP: LazyLock<FxHashMap<&'static str, Name>> = LazyLock::new(|| {
    let mut m = FxHashMap::default();
    for (i, s) in names::STRINGS.iter().enumerate() {
        // `i < STATIC_COUNT`, a u32.
        #[allow(clippy::cast_possible_truncation)]
        m.insert(*s, Name(i as u32));
    }
    m
});

/// The static [`Name`] for `s`, if it has one (the same in every document).
pub fn static_name(s: &str) -> Option<Name> {
    STATIC_LOOKUP.get(s).copied()
}

/// The names a document uses beyond the static ones.
#[derive(Clone, Debug, Default)]
struct DocNames {
    strings: Vec<Box<str>>,
    lookup: FxHashMap<Box<str>, Name>,
}

/// What kind of node: elements, and the element-like nodes lxml lists among the children.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Element,
    Comment,
    Pi,
    Entity,
}

/// One attribute, in document order on its element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attr {
    pub name: Name,
    pub value: String,
}

/// One node of the arena.
#[derive(Clone, Debug)]
pub struct Node {
    pub kind: Kind,
    /// Element name ([`names::NONE`] for comments and the like, whose lxml `.tag` is not a str).
    pub tag: Name,
    pub attrs: Vec<Attr>,
    /// lxml `.text`. For comments and PIs: their content (not part of `itertext()`).
    pub text: Option<String>,
    /// lxml `.tail`.
    pub tail: Option<String>,
    pub parent: NodeId,
    pub first: NodeId,
    pub last: NodeId,
    pub prev: NodeId,
    pub next: NodeId,
    /// Number of children (lxml `len(element)`).
    pub nchild: u32,
}

impl Node {
    fn new(kind: Kind, tag: Name) -> Self {
        Node {
            kind,
            tag,
            attrs: Vec::new(),
            text: None,
            tail: None,
            parent: NIL,
            first: NIL,
            last: NIL,
            prev: NIL,
            next: NIL,
            nchild: 0,
        }
    }
}

/// A set of element names (lxml's `_MultiTagMatcher` over tag names).
#[derive(Clone, Default, Debug)]
pub struct TagSet {
    bits: [u64; 8],
    extra: Vec<u32>,
}

impl TagSet {
    pub fn of(tags: &[Name]) -> Self {
        let mut s = TagSet::default();
        for &t in tags {
            s.insert(t);
        }
        s
    }

    pub fn insert(&mut self, t: Name) {
        if t.0 < 512 {
            self.bits[(t.0 / 64) as usize] |= 1 << (t.0 % 64);
        } else if !self.extra.contains(&t.0) {
            self.extra.push(t.0);
        }
    }

    #[inline]
    pub fn contains(&self, t: Name) -> bool {
        if t.0 < 512 {
            (self.bits[(t.0 / 64) as usize] >> (t.0 % 64)) & 1 == 1
        } else {
            self.extra.contains(&t.0)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bits == [0; 8] && self.extra.is_empty()
    }
}

/// What a tree walker stops on (lxml `iter()` arguments).
#[derive(Clone, Debug)]
pub enum Matcher {
    /// `iter()`: elements, comments, PIs, entities.
    AnyNode,
    /// `iter("*")`.
    AnyElement,
    Tags(TagSet),
}

impl Matcher {
    pub fn tags(tags: &[Name]) -> Self {
        Matcher::Tags(TagSet::of(tags))
    }

    #[inline]
    pub fn matches(&self, n: &Node) -> bool {
        match self {
            Matcher::AnyNode => true,
            Matcher::AnyElement => n.kind == Kind::Element,
            Matcher::Tags(s) => n.kind == Kind::Element && s.contains(n.tag),
        }
    }
}

/// Append `b` to the text run `a` the way adjacent libxml2 text nodes combine: `None` is "no
/// text node", so `None + None` stays `None` and `None + Some("")` becomes `Some("")`.
pub fn cat(a: &mut Option<String>, b: Option<&str>) {
    if let Some(b) = b {
        match a {
            Some(s) => s.push_str(b),
            None => *a = Some(b.to_owned()),
        }
    }
}

/// A parsed document: the arena, its names, and the element lxml returned.
#[derive(Clone, Debug)]
pub struct Document {
    pub nodes: Vec<Node>,
    /// The element `lxml.html.fromstring` returned. It can have a parent (lxml returns the
    /// single child of `<body>` for a one-element fragment): see [`Document::top`].
    pub root: NodeId,
    names: DocNames,
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl Document {
    /// An empty arena (`root` is [`NIL`]).
    pub fn new() -> Self {
        Document {
            nodes: Vec::new(),
            root: NIL,
            names: DocNames::default(),
        }
    }

    /// An empty arena with room for `n` nodes.
    pub fn with_capacity(n: usize) -> Self {
        Document {
            nodes: Vec::with_capacity(n),
            root: NIL,
            names: DocNames::default(),
        }
    }

    // ---- names ----------------------------------------------------------------------------

    /// The string of `name`.
    pub fn name(&self, name: Name) -> &str {
        if name.is_static() {
            names::STRINGS[name.0 as usize]
        } else {
            self.names
                .strings
                .get((name.0 - names::STATIC_COUNT) as usize)
                .map_or("", |s| s)
        }
    }

    /// The [`Name`] for `s` if this document has one (static or seen in this document).
    pub fn lookup(&self, s: &str) -> Option<Name> {
        static_name(s).or_else(|| self.names.lookup.get(s).copied())
    }

    /// The [`Name`] for `s`, adding it to this document if needed.
    pub fn intern(&mut self, s: &str) -> Name {
        if let Some(n) = self.lookup(s) {
            return n;
        }
        // Distinct names are bounded by the input size (a name is at least one byte).
        #[allow(clippy::cast_possible_truncation)]
        let n = Name(names::STATIC_COUNT + self.names.strings.len() as u32);
        self.names.strings.push(s.into());
        self.names.lookup.insert(s.into(), n);
        n
    }

    // ---- nodes ----------------------------------------------------------------------------

    /// Add a detached node and return its id.
    pub fn push(&mut self, kind: Kind, tag: Name) -> NodeId {
        // The parser creates at most one node per 3 input bytes, and inputs are below 2 GiB.
        #[allow(clippy::cast_possible_truncation)]
        let id = self.nodes.len() as NodeId;
        debug_assert!(id != NIL);
        self.nodes.push(Node::new(kind, tag));
        id
    }

    /// lxml `Element(tag)`: a new, detached element.
    pub fn element(&mut self, tag: Name) -> NodeId {
        self.push(Kind::Element, tag)
    }

    /// lxml `SubElement(parent, tag)`.
    pub fn sub_element(&mut self, parent: NodeId, tag: Name) -> NodeId {
        let e = self.element(tag);
        self.link_last(parent, e);
        e
    }

    #[inline]
    pub fn n(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    #[inline]
    pub fn nm(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    #[inline]
    pub fn tag(&self, id: NodeId) -> Name {
        self.nodes[id as usize].tag
    }

    #[inline]
    pub fn is_element(&self, id: NodeId) -> bool {
        self.nodes[id as usize].kind == Kind::Element
    }

    pub fn set_tag(&mut self, id: NodeId, tag: Name) {
        self.nodes[id as usize].tag = tag;
    }

    /// The element's tag as a string (`""` for comments and the like).
    pub fn tag_name(&self, id: NodeId) -> &str {
        self.name(self.tag(id))
    }

    #[inline]
    pub fn text(&self, id: NodeId) -> Option<&str> {
        self.nodes[id as usize].text.as_deref()
    }

    #[inline]
    pub fn tail(&self, id: NodeId) -> Option<&str> {
        self.nodes[id as usize].tail.as_deref()
    }

    pub fn set_text(&mut self, id: NodeId, v: Option<String>) {
        self.nodes[id as usize].text = v;
    }

    pub fn set_tail(&mut self, id: NodeId, v: Option<String>) {
        self.nodes[id as usize].tail = v;
    }

    // ---- attributes -----------------------------------------------------------------------

    /// lxml `get(name)`.
    pub fn get(&self, id: NodeId, name: &str) -> Option<&str> {
        self.lookup(name).and_then(|n| self.attr(id, n))
    }

    /// lxml `get(name)` by interned name.
    pub fn attr(&self, id: NodeId, name: Name) -> Option<&str> {
        self.nodes[id as usize]
            .attrs
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.value.as_str())
    }

    /// lxml `set(name, value)`: replaces the value in place or appends the attribute.
    pub fn set(&mut self, id: NodeId, name: &str, value: &str) {
        let n = self.intern(name);
        self.set_attr(id, n, value);
    }

    /// lxml `set(name, value)` by interned name.
    pub fn set_attr(&mut self, id: NodeId, name: Name, value: &str) {
        let a = &mut self.nodes[id as usize].attrs;
        match a.iter_mut().find(|a| a.name == name) {
            Some(a) => value.clone_into(&mut a.value),
            None => a.push(Attr {
                name,
                value: value.to_owned(),
            }),
        }
    }

    /// lxml `attrib.clear()`.
    pub fn attrib_clear(&mut self, id: NodeId) {
        self.nodes[id as usize].attrs.clear();
    }

    /// lxml `attrib.items()`: names and values in document order.
    pub fn attrs(&self, id: NodeId) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.nodes[id as usize]
            .attrs
            .iter()
            .map(|a| (self.name(a.name), a.value.as_str()))
    }

    // ---- navigation -----------------------------------------------------------------------

    /// lxml `getparent()` (`None` for a top-level element: the document node is not an element).
    #[inline]
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        some(self.nodes[id as usize].parent)
    }

    /// lxml `getnext()`.
    #[inline]
    pub fn next(&self, id: NodeId) -> Option<NodeId> {
        some(self.nodes[id as usize].next)
    }

    /// lxml `getprevious()`.
    #[inline]
    pub fn prev(&self, id: NodeId) -> Option<NodeId> {
        some(self.nodes[id as usize].prev)
    }

    /// lxml `len(element)`.
    #[inline]
    pub fn len(&self, id: NodeId) -> usize {
        self.nodes[id as usize].nchild as usize
    }

    /// lxml `element[-1]`.
    pub fn last_child(&self, id: NodeId) -> Option<NodeId> {
        some(self.nodes[id as usize].last)
    }

    /// lxml `element[0]`.
    pub fn first_child(&self, id: NodeId) -> Option<NodeId> {
        some(self.nodes[id as usize].first)
    }

    /// lxml `element[i]`, negative indices counting from the end.
    pub fn child_at(&self, id: NodeId, i: isize) -> Option<NodeId> {
        let n = isize::try_from(self.len(id)).ok()?;
        let i = if i < 0 { n + i } else { i };
        if i < 0 || i >= n {
            return None;
        }
        self.children(id).nth(usize::try_from(i).ok()?)
    }

    /// `for c in element` (all element-like children). Callers must not unlink while iterating.
    pub fn children(&self, id: NodeId) -> Children<'_> {
        Children {
            d: self,
            cur: self.nodes[id as usize].first,
        }
    }

    pub fn children_vec(&self, id: NodeId) -> Vec<NodeId> {
        self.children(id).collect()
    }

    /// The top-most ancestor (inclusive): stands for lxml's `getroottree().getroot()`.
    pub fn top(&self, mut id: NodeId) -> NodeId {
        while let Some(p) = self.parent(id) {
            id = p;
        }
        id
    }

    /// `iterancestors()`.
    pub fn ancestors(&self, id: NodeId) -> Ancestors<'_> {
        Ancestors {
            d: self,
            cur: self.nodes[id as usize].parent,
        }
    }

    /// The nearest ancestor whose tag is in `set`.
    pub fn first_ancestor_in(&self, id: NodeId, set: &TagSet) -> Option<NodeId> {
        self.ancestors(id)
            .find(|&a| self.is_element(a) && set.contains(self.tag(a)))
    }

    pub fn is_ancestor_or_self(&self, anc: NodeId, mut id: NodeId) -> bool {
        loop {
            if id == anc {
                return true;
            }
            match self.parent(id) {
                Some(p) => id = p,
                None => return false,
            }
        }
    }

    // ---- structure ------------------------------------------------------------------------

    /// Link the detached node `child` as the last child of `parent`.
    pub(crate) fn link_last(&mut self, parent: NodeId, child: NodeId) {
        let last = self.nodes[parent as usize].last;
        {
            let c = &mut self.nodes[child as usize];
            c.parent = parent;
            c.prev = last;
            c.next = NIL;
        }
        if last == NIL {
            self.nodes[parent as usize].first = child;
        } else {
            self.nodes[last as usize].next = child;
        }
        let p = &mut self.nodes[parent as usize];
        p.last = child;
        p.nchild += 1;
    }

    fn link_before(&mut self, sibling: NodeId, child: NodeId) {
        let parent = self.nodes[sibling as usize].parent;
        let prev = self.nodes[sibling as usize].prev;
        {
            let c = &mut self.nodes[child as usize];
            c.parent = parent;
            c.prev = prev;
            c.next = sibling;
        }
        self.nodes[sibling as usize].prev = child;
        if prev == NIL {
            self.nodes[parent as usize].first = child;
        } else {
            self.nodes[prev as usize].next = child;
        }
        self.nodes[parent as usize].nchild += 1;
    }

    /// Detach a node from its parent; its tail stays with it (lxml `remove`).
    pub fn unlink(&mut self, id: NodeId) {
        let (parent, prev, next) = {
            let n = &self.nodes[id as usize];
            (n.parent, n.prev, n.next)
        };
        if parent == NIL {
            return;
        }
        if prev == NIL {
            self.nodes[parent as usize].first = next;
        } else {
            self.nodes[prev as usize].next = next;
        }
        if next == NIL {
            self.nodes[parent as usize].last = prev;
        } else {
            self.nodes[next as usize].prev = prev;
        }
        self.nodes[parent as usize].nchild -= 1;
        let n = &mut self.nodes[id as usize];
        n.parent = NIL;
        n.prev = NIL;
        n.next = NIL;
    }

    /// lxml `parent.append(child)`: moves the child (and its tail) from wherever it is. Does
    /// nothing where lxml raises "cannot append parent to itself".
    pub fn append(&mut self, parent: NodeId, child: NodeId) {
        if self.is_ancestor_or_self(child, parent) {
            debug_assert!(false, "append would create a cycle");
            return;
        }
        self.unlink(child);
        self.link_last(parent, child);
    }

    /// lxml `parent.insert(index, child)` (etree.pyx `_Element.insert`): the child at `index`
    /// is found with `_findChild` (a negative index counts from the end, `-1` being the last
    /// child), and an index past either end appends; `child` moves before that child with its
    /// tail. Unlike Python's `list.insert`, `insert(-k)` with `k > len` appends. When `child`
    /// is that child, libxml2's `xmlAddPrevSibling` does nothing, and neither does this. Does
    /// nothing where lxml raises "cannot append parent to itself".
    pub fn insert(&mut self, parent: NodeId, index: isize, child: NodeId) {
        let n = self.len(parent);
        let at = if index < 0 {
            // _findChildBackwards(c_node, -index - 1)
            usize::try_from(-(index + 1))
                .ok()
                .filter(|&k| k < n)
                .and_then(|k| self.children(parent).nth(n - 1 - k))
        } else {
            usize::try_from(index)
                .ok()
                .filter(|&k| k < n)
                .and_then(|k| self.children(parent).nth(k))
        };
        match at {
            None => self.append(parent, child),
            Some(s) if s == child => {}
            Some(s) => {
                if self.is_ancestor_or_self(child, parent) {
                    debug_assert!(false, "insert would create a cycle");
                    return;
                }
                self.unlink(child);
                self.link_before(s, child);
            }
        }
    }

    /// Text slot preceding a child position: the previous sibling's tail or the parent's text.
    fn preceding_text_mut(&mut self, parent: NodeId, prev: NodeId) -> &mut Option<String> {
        if prev == NIL {
            &mut self.nodes[parent as usize].text
        } else {
            &mut self.nodes[prev as usize].tail
        }
    }

    /// lxml `_replaceNodeByChildren` (strip_tags): splice text, children and tail into the
    /// parent.
    fn replace_by_children(&mut self, x: NodeId) {
        let (parent, prev, next, first, last) = {
            let n = &self.nodes[x as usize];
            (n.parent, n.prev, n.next, n.first, n.last)
        };
        if parent == NIL {
            return;
        }
        let text = self.nodes[x as usize].text.take();
        let tail = self.nodes[x as usize].tail.take();
        if first == NIL {
            let slot = self.preceding_text_mut(parent, prev);
            cat(slot, text.as_deref());
            cat(slot, tail.as_deref());
            self.unlink(x);
            return;
        }
        cat(self.preceding_text_mut(parent, prev), text.as_deref());
        cat(&mut self.nodes[last as usize].tail, tail.as_deref());
        let k = self.nodes[x as usize].nchild;
        let mut c = first;
        while c != NIL {
            self.nodes[c as usize].parent = parent;
            c = self.nodes[c as usize].next;
        }
        if prev == NIL {
            self.nodes[parent as usize].first = first;
        } else {
            self.nodes[prev as usize].next = first;
        }
        self.nodes[first as usize].prev = prev;
        if next == NIL {
            self.nodes[parent as usize].last = last;
        } else {
            self.nodes[next as usize].prev = last;
        }
        self.nodes[last as usize].next = next;
        self.nodes[parent as usize].nchild += k - 1;
        let n = &mut self.nodes[x as usize];
        n.first = NIL;
        n.last = NIL;
        n.nchild = 0;
        n.parent = NIL;
        n.prev = NIL;
        n.next = NIL;
    }

    /// lxml `etree.strip_tags(root, *tags)`, replaying `_strip_tags`'s walk.
    pub fn strip_tags(&mut self, root: NodeId, m: &Matcher) {
        if let Matcher::Tags(s) = m {
            if s.is_empty() {
                return;
            }
        }
        let mut c_node = root;
        // BEGIN_FOR_EACH_ELEMENT_FROM(root, root, 1)
        while c_node != NIL {
            if self.is_element(c_node) {
                let mut c_child = self.nodes[c_node as usize].first;
                while c_child != NIL {
                    if !m.matches(&self.nodes[c_child as usize]) {
                        c_child = self.nodes[c_child as usize].next;
                        continue;
                    }
                    if self.is_element(c_child) {
                        let f = self.nodes[c_child as usize].first;
                        let c_next = if f == NIL {
                            self.nodes[c_child as usize].next
                        } else {
                            f
                        };
                        self.replace_by_children(c_child);
                        c_child = c_next;
                    } else {
                        // A comment or PI: unlinked without its tail, which stays in place.
                        let c_next = self.nodes[c_child as usize].next;
                        let tail = self.nodes[c_child as usize].tail.take();
                        let prev = self.nodes[c_child as usize].prev;
                        cat(self.preceding_text_mut(c_node, prev), tail.as_deref());
                        self.unlink(c_child);
                        c_child = c_next;
                    }
                }
            }
            c_node = self.traverse_next(root, c_node);
        }
    }

    /// lxml `etree.strip_elements(root, *tags, with_tail=True)`.
    pub fn strip_elements(&mut self, root: NodeId, m: &Matcher) {
        let mut c_node = root;
        while c_node != NIL {
            if self.is_element(c_node) {
                let mut c_child = self.nodes[c_node as usize].first;
                while c_child != NIL {
                    let c_next = self.nodes[c_child as usize].next;
                    if m.matches(&self.nodes[c_child as usize]) {
                        self.unlink(c_child);
                    }
                    c_child = c_next;
                }
            }
            c_node = self.traverse_next(root, c_node);
        }
    }

    /// A detached copy of node `id` of `src` (kind, tag, attributes, text and tail), with names
    /// re-interned when `src` is another document.
    fn copy_node_from(&mut self, src: Option<&Document>, id: NodeId) -> NodeId {
        let n = match src {
            Some(s) => &s.nodes[id as usize],
            None => &self.nodes[id as usize],
        };
        let (kind, tag) = (n.kind, n.tag);
        let mut attrs = n.attrs.clone();
        let text = n.text.clone();
        let tail = n.tail.clone();
        let tag = match src {
            Some(s) if !tag.is_static() => self.intern(s.name(tag)),
            _ => tag,
        };
        if let Some(s) = src {
            for a in &mut attrs {
                if !a.name.is_static() {
                    a.name = self.intern(s.name(a.name));
                }
            }
        }
        let me = self.push(kind, tag);
        let m = &mut self.nodes[me as usize];
        m.attrs = attrs;
        m.text = text;
        m.tail = tail;
        me
    }

    fn copy_subtree(&mut self, src: Option<&Document>, id: NodeId) -> NodeId {
        let first_of = |d: &Document, x: NodeId| d.nodes[x as usize].first;
        let next_of = |d: &Document, x: NodeId| d.nodes[x as usize].next;
        let me = self.copy_node_from(src, id);
        let f = match src {
            Some(s) => first_of(s, id),
            None => first_of(self, id),
        };
        // (next source child to copy, the copy it goes under)
        let mut stack: Vec<(NodeId, NodeId)> = vec![(f, me)];
        while let Some(&(c, under)) = stack.last() {
            if c == NIL {
                stack.pop();
                continue;
            }
            let (nx, cf) = match src {
                Some(s) => (next_of(s, c), first_of(s, c)),
                None => (next_of(self, c), first_of(self, c)),
            };
            if let Some(top) = stack.last_mut() {
                top.0 = nx;
            }
            let cc = self.copy_node_from(src, c);
            self.link_last(under, cc);
            if cf != NIL {
                stack.push((cf, cc));
            }
        }
        me
    }

    /// lxml `copy(element)` / `deepcopy(element)`: the subtree and the element's tail, as a new
    /// detached root in this arena.
    pub fn deep_copy(&mut self, id: NodeId) -> NodeId {
        self.copy_subtree(None, id)
    }

    /// Copy the subtree under `id` of another document into this arena (detached; the root
    /// keeps its tail), re-interning names that are not static.
    pub fn import(&mut self, src: &Document, id: NodeId) -> NodeId {
        self.copy_subtree(Some(src), id)
    }

    // ---- walkers --------------------------------------------------------------------------

    /// `_LX__TRAVERSE_TO_NEXT` (etree_defs.h): the next node in document order below `stop`.
    #[inline]
    pub fn traverse_next(&self, stop: NodeId, mut c: NodeId) -> NodeId {
        let f = self.nodes[c as usize].first;
        if f != NIL {
            return f;
        }
        if c == stop {
            return NIL;
        }
        let mut nx = self.nodes[c as usize].next;
        while nx == NIL {
            c = self.nodes[c as usize].parent;
            if c == NIL || c == stop {
                return NIL;
            }
            nx = self.nodes[c as usize].next;
        }
        nx
    }

    /// Snapshot of `iter(...)` / `iterdescendants(...)` for loops that do not change the
    /// structure.
    pub fn iter_vec(&self, top: NodeId, m: &Matcher, inclusive: bool) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut it = Iter::new(self, top, m.clone(), inclusive);
        while let Some(x) = it.next(self) {
            out.push(x);
        }
        out
    }

    /// lxml `itertext()`: element texts and the tails of everything below `id` (not comment
    /// content, not the tail of `id` itself), in document order.
    pub fn itertext(&self, id: NodeId, f: &mut dyn FnMut(&str)) {
        let n = &self.nodes[id as usize];
        if n.kind != Kind::Element {
            return;
        }
        if let Some(s) = &n.text {
            f(s);
        }
        let mut c = n.first;
        while c != NIL {
            let cn = &self.nodes[c as usize];
            if cn.kind == Kind::Element {
                if let Some(s) = &cn.text {
                    f(s);
                }
                if cn.first != NIL {
                    c = cn.first;
                    continue;
                }
            }
            // `c` is finished: its tail, then the next sibling, climbing out of finished parents.
            loop {
                let cn = &self.nodes[c as usize];
                if let Some(s) = &cn.tail {
                    f(s);
                }
                if cn.next != NIL {
                    c = cn.next;
                    break;
                }
                c = cn.parent;
                if c == id || c == NIL {
                    return;
                }
            }
        }
    }

    /// `sep.join(element.itertext())`.
    pub fn itertext_join(&self, id: NodeId, sep: &str) -> String {
        let mut out = String::new();
        let mut first = true;
        self.itertext(id, &mut |s| {
            if !first {
                out.push_str(sep);
            }
            first = false;
            out.push_str(s);
        });
        out
    }

    /// lxml.html `text_content()` (XPath `string()`): all descendant text, without the tail.
    pub fn text_content(&self, id: NodeId) -> String {
        self.itertext_join(id, "")
    }

    /// Append the subtree under `id` to `out` in the `.tree` line format (one JSON array per
    /// line, as `json.dumps(..., ensure_ascii=False)` writes it): `["S", tag]`, then
    /// `["A", name, value]` per attribute, `["T", text]` if the text is not empty, the
    /// children, `["E"]`, and `["L", tail]` if the tail is not empty and `id` is not `root`.
    /// Non-elements are `["X", class, text]` with lxml.html's class names.
    pub fn dump(&self, id: NodeId, root: bool, out: &mut String) {
        let open = |d: &Document, x: NodeId, out: &mut String| {
            let n = &d.nodes[x as usize];
            if n.kind == Kind::Element {
                line(out, &["S", d.name(n.tag)]);
                for a in &n.attrs {
                    line(out, &["A", d.name(a.name), &a.value]);
                }
                if let Some(t) = n.text.as_deref().filter(|t| !t.is_empty()) {
                    line(out, &["T", t]);
                }
                n.first != NIL
            } else {
                let class = match n.kind {
                    Kind::Comment => "HtmlComment",
                    Kind::Pi => "HtmlProcessingInstruction",
                    _ => "HtmlEntity",
                };
                line(out, &["X", class, n.text.as_deref().unwrap_or("")]);
                false
            }
        };
        let close = |d: &Document, x: NodeId, out: &mut String| {
            let n = &d.nodes[x as usize];
            if n.kind == Kind::Element {
                line(out, &["E"]);
            }
            if x != id || !root {
                if let Some(t) = n.tail.as_deref().filter(|t| !t.is_empty()) {
                    line(out, &["L", t]);
                }
            }
        };
        let mut c = id;
        loop {
            if open(self, c, out) {
                c = self.nodes[c as usize].first;
                continue;
            }
            loop {
                close(self, c, out);
                if c == id {
                    return;
                }
                let n = &self.nodes[c as usize];
                if n.next != NIL {
                    c = n.next;
                    break;
                }
                c = n.parent;
            }
        }
    }
}

#[inline]
fn some(id: NodeId) -> Option<NodeId> {
    (id != NIL).then_some(id)
}

/// One `.tree` line: a JSON array of strings, escaped as Python's `json.dumps` with
/// `ensure_ascii=False` escapes them.
fn line(out: &mut String, parts: &[&str]) {
    out.push('[');
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('"');
        for c in p.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if c < ' ' => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push_str("]\n");
}

/// The children of a node (see [`Document::children`]).
#[derive(Debug)]
pub struct Children<'a> {
    d: &'a Document,
    cur: NodeId,
}

impl Iterator for Children<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        if self.cur == NIL {
            return None;
        }
        let c = self.cur;
        self.cur = self.d.nodes[c as usize].next;
        Some(c)
    }
}

/// The ancestors of a node, nearest first (see [`Document::ancestors`]).
#[derive(Debug)]
pub struct Ancestors<'a> {
    d: &'a Document,
    cur: NodeId,
}

impl Iterator for Ancestors<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        if self.cur == NIL {
            return None;
        }
        let c = self.cur;
        self.cur = self.d.nodes[c as usize].parent;
        Some(c)
    }
}

/// lxml `ElementDepthFirstIterator`: holds the next match, computed when the current one is
/// returned. It borrows the document only per step, so the loop body may edit the tree.
#[derive(Debug)]
pub struct Iter {
    top: NodeId,
    next: NodeId,
    m: Matcher,
}

impl Iter {
    pub fn new(d: &Document, top: NodeId, m: Matcher, inclusive: bool) -> Self {
        let mut it = Iter { top, next: top, m };
        if !inclusive || !it.m.matches(d.n(top)) {
            it.next(d);
        }
        it
    }

    pub fn next(&mut self, d: &Document) -> Option<NodeId> {
        let cur = self.next;
        if cur == NIL {
            return None;
        }
        let mut c = cur;
        loop {
            c = d.traverse_next(self.top, c);
            if c == NIL || self.m.matches(d.n(c)) {
                break;
            }
        }
        self.next = c;
        Some(cur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Document, NodeId, NodeId, NodeId) {
        // <div>a<p>b<b>c</b>d</p>e<span/>f</div>
        let mut d = Document::new();
        let div = d.element(names::DIV);
        d.set_text(div, Some("a".into()));
        let p = d.sub_element(div, names::P);
        d.set_text(p, Some("b".into()));
        let b = d.sub_element(p, names::B);
        d.set_text(b, Some("c".into()));
        d.set_tail(b, Some("d".into()));
        d.set_tail(p, Some("e".into()));
        let span = d.sub_element(div, names::SPAN);
        d.set_tail(span, Some("f".into()));
        d.root = div;
        (d, div, p, b)
    }

    #[test]
    fn static_names_have_fixed_ids() {
        assert_eq!(static_name("div"), Some(names::DIV));
        assert_eq!(static_name(""), Some(names::NONE));
        assert_eq!(static_name("xyz"), None);
        let mut seen = std::collections::HashSet::new();
        for s in names::STRINGS {
            assert!(seen.insert(s), "duplicate static name {s:?}");
        }
    }

    #[test]
    fn interning() {
        let mut d = Document::new();
        let x = d.intern("fb:like");
        assert!(!x.is_static());
        assert_eq!(d.intern("fb:like"), x);
        assert_eq!(d.name(x), "fb:like");
        assert_eq!(d.intern("p"), names::P);
        assert_eq!(d.lookup("nope"), None);
    }

    #[test]
    fn text_walks() {
        let (d, div, p, _) = sample();
        assert_eq!(d.text_content(div), "abcdef");
        assert_eq!(d.text_content(p), "bcd");
        assert_eq!(d.itertext_join(div, "|"), "a|b|c|d|e|f");
        let mut s = String::new();
        d.dump(div, true, &mut s);
        assert_eq!(
            s,
            "[\"S\", \"div\"]\n[\"T\", \"a\"]\n[\"S\", \"p\"]\n[\"T\", \"b\"]\n[\"S\", \"b\"]\n\
             [\"T\", \"c\"]\n[\"E\"]\n[\"L\", \"d\"]\n[\"E\"]\n[\"L\", \"e\"]\n[\"S\", \"span\"]\n\
             [\"E\"]\n[\"L\", \"f\"]\n[\"E\"]\n"
        );
    }

    #[test]
    fn json_escapes_match_python() {
        let mut s = String::new();
        line(&mut s, &["T", "a\"\\\n\r\t\u{8}\u{c}\u{1}\u{7f}\u{2028}é"]);
        assert_eq!(
            s,
            "[\"T\", \"a\\\"\\\\\\n\\r\\t\\b\\f\\u0001\u{7f}\u{2028}é\"]\n"
        );
    }

    #[test]
    fn copy_and_import() {
        let (mut d, div, p, _) = sample();
        let copy = d.deep_copy(p);
        assert_eq!(d.parent(copy), None);
        assert_eq!(d.tail(copy), Some("e"));
        assert_eq!(d.text_content(copy), "bcd");
        let mut other = Document::new();
        let custom = other.intern("custom-el");
        let el = other.element(custom);
        other.set(el, "data-foo", "1");
        let bold = other.sub_element(el, names::B);
        other.set_text(bold, Some("z".into()));
        let imported = d.import(&other, el);
        d.append(div, imported);
        assert_eq!(d.tag_name(imported), "custom-el");
        assert_eq!(d.get(imported, "data-foo"), Some("1"));
        assert_eq!(d.text_content(div), "abcdefz");
    }

    #[test]
    fn strip_tags_splices() {
        let (mut d, div, _, _) = sample();
        d.strip_tags(div, &Matcher::tags(&[names::P]));
        assert_eq!(d.text(div), Some("ab"));
        assert_eq!(d.text_content(div), "abcdef");
        assert_eq!(d.len(div), 2);
        let b = d.first_child(div).unwrap();
        assert_eq!(d.tag(b), names::B);
        assert_eq!(d.tail(b), Some("de"));
    }

    #[test]
    fn insert_and_append() {
        let (mut d, div, p, _) = sample();
        let span = d.last_child(div).unwrap();
        d.insert(div, 0, span);
        assert_eq!(d.children_vec(div), vec![span, p]);
        d.insert(div, 0, span);
        assert_eq!(d.children_vec(div), vec![span, p]);
        d.insert(div, -1, span);
        assert_eq!(d.children_vec(div), vec![span, p]);
        d.append(div, span);
        assert_eq!(d.children_vec(div), vec![p, span]);
        assert_eq!(d.child_at(div, -1), Some(span));
        assert_eq!(d.child_at(div, 2), None);
        // lxml: `insert(-k)` past the front appends; `insert(-1)` goes before the last child.
        let em = d.element(names::EM);
        d.insert(div, -5, em);
        assert_eq!(d.children_vec(div), vec![p, span, em]);
        d.insert(div, -1, span);
        assert_eq!(d.children_vec(div), vec![p, span, em]);
        d.insert(div, -2, em);
        assert_eq!(d.children_vec(div), vec![p, em, span]);
        // Moving keeps the tail with the element.
        assert_eq!(d.tail(span), Some("f"));
        assert_eq!(d.text_content(div), "abcdef");
    }

    #[test]
    fn deep_trees_do_not_recurse() {
        let mut d = Document::new();
        let root = d.element(names::DIV);
        let mut cur = root;
        for _ in 0..200_000 {
            cur = d.sub_element(cur, names::DIV);
            d.set_tail(cur, Some("t".into()));
        }
        d.set_text(cur, Some("x".into()));
        d.root = root;
        assert_eq!(d.text_content(root).len(), 200_001);
        let c = d.deep_copy(root);
        let mut s = String::new();
        d.dump(c, true, &mut s);
        assert_eq!(s.lines().count(), 2 + 3 * 200_000 + 1);
        drop(d);
    }

    #[test]
    fn iter_is_lxml_depth_first() {
        let (d, div, p, b) = sample();
        let span = d.last_child(div).unwrap();
        assert_eq!(
            d.iter_vec(div, &Matcher::AnyElement, true),
            vec![div, p, b, span]
        );
        assert_eq!(
            d.iter_vec(div, &Matcher::tags(&[names::B, names::SPAN]), false),
            vec![b, span]
        );
    }
}
