use super::*;

fn dict() -> &'static Dictionary {
    Dictionary::default_ref()
}

fn cut(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    dict()
        .cut(s, true, &ForceSplit::default(), &mut out)
        .unwrap();
    out
}

fn cut_no_hmm(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    dict()
        .cut(s, false, &ForceSplit::default(), &mut out)
        .unwrap();
    out
}

fn cut_all(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    dict().cut_all(s, &mut out);
    out.into_iter().map(Cow::into_owned).collect()
}

fn search(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    dict()
        .cut_for_search(s, true, &ForceSplit::default(), &mut out)
        .unwrap();
    out
}

// The examples of jieba's README (outputs of jieba 0.42.1).
#[test]
fn readme_examples() {
    assert_eq!(
        cut_all("我来到北京清华大学"),
        ["我", "来到", "北京", "清华", "清华大学", "华大", "大学"]
    );
    assert_eq!(
        cut("我来到北京清华大学"),
        ["我", "来到", "北京", "清华大学"]
    );
    assert_eq!(
        cut("他来到了网易杭研大厦"),
        ["他", "来到", "了", "网易", "杭研", "大厦"]
    );
    assert_eq!(
        search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造"),
        [
            "小明",
            "硕士",
            "毕业",
            "于",
            "中国",
            "科学",
            "学院",
            "科学院",
            "中国科学院",
            "计算",
            "计算所",
            "，",
            "后",
            "在",
            "日本",
            "京都",
            "大学",
            "日本京都大学",
            "深造"
        ]
    );
}

#[test]
fn blocks_and_separators() {
    assert_eq!(cut("a， b  c"), ["a", "，", " ", "b", " ", " ", "c"]);
    assert_eq!(
        cut_all("a， b  c"),
        ["a", "，", " ", "", "b", "", " ", "", " ", "", "c"]
    );
    assert_eq!(cut("x\r\ny\n\rz"), ["x", "\r\n", "y", "\n", "\r", "z"]);
    assert_eq!(cut("").len(), 0);
    assert_eq!(cut_all("").len(), 0);
}

#[test]
fn every_mode_covers_the_text() {
    for s in [
        "我来到北京清华大学",
        "IS_ALPHA 中文 C++ 3.14% AT&T",
        "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
        "  前导空格\r\n后面😀 emoji 繁體中文字",
    ] {
        assert_eq!(cut(s).concat(), s);
        assert_eq!(cut_no_hmm(s).concat(), s);
    }
}

#[test]
fn tokenize_offsets() {
    let mut out = Vec::new();
    dict()
        .tokenize(
            "永和服装饰品有限公司",
            false,
            true,
            &ForceSplit::default(),
            &mut out,
        )
        .unwrap();
    assert_eq!(
        out,
        [
            ("永和", 0, 2),
            ("服装", 2, 4),
            ("饰品", 4, 6),
            ("有限公司", 6, 10)
        ]
    );
    out.clear();
    dict()
        .tokenize(
            "永和服装饰品有限公司",
            true,
            true,
            &ForceSplit::default(),
            &mut out,
        )
        .unwrap();
    assert_eq!(
        out,
        [
            ("永和", 0, 2),
            ("服装", 2, 4),
            ("饰品", 4, 6),
            ("有限", 6, 8),
            ("公司", 8, 10),
            ("有限公司", 6, 10)
        ]
    );
}

#[test]
fn user_words_and_force_split() {
    let mut d = dict().clone();
    let mut fs = ForceSplit::default();
    let mut out = Vec::new();
    d.cut("李小福是创新办主任", true, &fs, &mut out).unwrap();
    assert_eq!(out, ["李小福", "是", "创新", "办", "主任"]);
    d.add_word("创新办", 3).unwrap();
    out.clear();
    d.cut("李小福是创新办主任", true, &fs, &mut out).unwrap();
    assert_eq!(out, ["李小福", "是", "创新办", "主任"]);
    // del_word: frequency 0 and a force split of the HMM's word.
    d.add_word("杭研", 0).unwrap();
    fs.insert("杭研".into());
    out.clear();
    d.cut("他来到了网易杭研大厦", true, &fs, &mut out).unwrap();
    assert_eq!(out, ["他", "来到", "了", "网易", "杭", "研", "大厦"]);
}

#[test]
fn math_domain_errors() {
    let empty = Dictionary::from_entries(Vec::new(), 0).unwrap();
    let mut out = Vec::new();
    // No word block: no calc, no error.
    empty
        .cut("，", true, &ForceSplit::default(), &mut out)
        .unwrap();
    assert_eq!(out, ["，"]);
    assert_eq!(
        empty.cut("中文", true, &ForceSplit::default(), &mut out),
        Err(CutError::MathDomain)
    );
    let mut d = dict().clone();
    d.add_word("中文", -5).unwrap();
    assert_eq!(
        d.cut("中文", true, &ForceSplit::default(), &mut out),
        Err(CutError::MathDomain)
    );
}

#[test]
fn suggest_freq_values() {
    let d = dict();
    // jieba 0.42.1: suggest_freq("清华大学") == 923 (FREQ 922).
    assert_eq!(d.get("清华大学"), Some(922));
    assert_eq!(d.suggest_freq("清华大学").unwrap(), 923);
    assert_eq!(d.suggest_freq("").unwrap(), DEFAULT_TOTAL + 1);
    assert_eq!(d.suggest_freq_split(&[]).unwrap(), 0);
}

/// Rebuild `data/dict.trie` from jieba 0.42.1's `dict.txt` (the path in
/// `JIEBA_DICT_TXT`; see `generate.py`), or check it against that file.
#[test]
#[ignore = "needs JIEBA_DICT_TXT; writes data/dict.trie"]
fn regenerate() {
    let path = std::env::var("JIEBA_DICT_TXT").expect("JIEBA_DICT_TXT");
    let blob = build_default_trie(&std::fs::read(path).expect("dict.txt"));
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/src/jieba/data/dict.trie");
    std::fs::write(out, &blob).expect("write dict.trie");
}

#[test]
fn bundled_trie_is_dict_txt() {
    let Ok(path) = std::env::var("JIEBA_DICT_TXT") else {
        return;
    };
    let blob = build_default_trie(&std::fs::read(path).expect("dict.txt"));
    assert!(
        blob == Dictionary::default_trie_bytes(),
        "data/dict.trie is stale: run the regenerate test"
    );
}

/// `gen_pfdict` over `dict.txt`, strictly: `word freq tag` on every line.
fn build_default_trie(txt: &[u8]) -> Vec<u8> {
    let txt = std::str::from_utf8(txt).expect("UTF-8");
    let mut words: Vec<(&str, i64)> = Vec::new();
    let mut total = 0i64;
    for line in txt.lines() {
        let line = line.trim_matches(|c: char| c.is_ascii_whitespace());
        let mut parts = line.split(' ');
        let word = parts.next().expect("word");
        let freq: i64 = parts.next().expect("freq").parse().expect("ASCII digits");
        words.push((word, freq));
        total += freq;
    }
    assert_eq!(words.len(), 349_046);
    assert_eq!(total, DEFAULT_TOTAL);
    Dictionary::serialize_words(&words).expect("the default dictionary fits the format")
}
