//! The PEP 440 form of a Cargo (SemVer) version, as maturin writes it into a wheel.
//!
//! maturin 1.16 reads the `Cargo.toml` version with `pep440_rs` 0.7.3's lenient parser (PEP
//! 440's normalization rules) and writes the normalized form into the wheel metadata, so
//! `1.0.0-dev` is `1.0.0.dev0` there. [`pep440`] is that conversion for the versions Cargo
//! accepts: `pyroid.__version__` reports what `importlib.metadata.version("pyroid")` does.

/// `version`, a Cargo version (`MAJOR.MINOR.PATCH[-PRE][+BUILD]`), in PEP 440's normalized form;
/// `None` if PEP 440 can't read it (maturin refuses to build such a version).
///
/// `-PRE` is read as PEP 440's optional pre-, post- and dev-release parts, in that order, each
/// label optionally preceded and followed by `.`, `-` or `_`, case-insensitive, with a missing
/// number meaning 0:
///
/// - `alpha`/`a` becomes `aN`, `beta`/`b` `bN`, `rc`/`c`/`pre`/`preview` `rcN`;
/// - `post`/`rev`/`r`, or a bare `-N`, becomes `.postN`;
/// - `dev` becomes `.devN`.
///
/// `+BUILD` is the local version: lowercased, `.`-separated, numbers without leading zeros.
///
/// ```
/// use pyroid_core::version::pep440;
///
/// assert_eq!(pep440("1.0.0-dev").as_deref(), Some("1.0.0.dev0"));
/// assert_eq!(pep440("1.2.3-rc.1+Build-07").as_deref(), Some("1.2.3rc1+build.7"));
/// assert_eq!(pep440("1.0.0-nightly"), None);
/// ```
pub fn pep440(version: &str) -> Option<String> {
    Parser {
        v: version.as_bytes(),
        i: 0,
    }
    .parse()
    .ok()
}

/// Not a PEP 440 version.
struct Invalid;

/// `pep440_rs`'s `Parser`, minus what a Cargo version can't contain (an epoch, a leading `v`,
/// whitespace, a `.*` wildcard).
struct Parser<'a> {
    v: &'a [u8],
    i: usize,
}

/// Pre-release spellings, matched in this order (`preview` before `pre`), and their PEP 440 kind.
const PRE: [(&str, &str); 8] = [
    ("alpha", "a"),
    ("beta", "b"),
    ("preview", "rc"),
    ("pre", "rc"),
    ("rc", "rc"),
    ("a", "a"),
    ("b", "b"),
    ("c", "rc"),
];

impl Parser<'_> {
    fn parse(mut self) -> Result<String, Invalid> {
        let mut release = vec![self.number()?.ok_or(Invalid)?.to_string()];
        loop {
            let at = self.i;
            if !self.bump(".") {
                break;
            }
            let Some(n) = self.number()? else {
                self.i = at;
                break;
            };
            release.push(n.to_string());
        }
        let pre = self.pre()?;
        let post = self.post()?;
        let dev = self.dev()?;
        let local = self.local()?;
        if self.i != self.v.len() {
            return Err(Invalid);
        }
        let release = release.join(".");
        let pre = pre
            .map(|(kind, n)| format!("{kind}{n}"))
            .unwrap_or_default();
        let post = post.map(|n| format!(".post{n}")).unwrap_or_default();
        let dev = dev.map(|n| format!(".dev{n}")).unwrap_or_default();
        let local = if local.is_empty() {
            String::new()
        } else {
            format!("+{}", local.join("."))
        };
        Ok(format!("{release}{pre}{post}{dev}{local}"))
    }

    fn pre(&mut self) -> Result<Option<(&'static str, u64)>, Invalid> {
        let at = self.i;
        self.separator();
        let Some(kind) = PRE
            .iter()
            .find_map(|&(spelling, kind)| self.bump(spelling).then_some(kind))
        else {
            self.i = at;
            return Ok(None);
        };
        self.separator();
        Ok(Some((kind, self.number()?.unwrap_or(0))))
    }

    fn post(&mut self) -> Result<Option<u64>, Invalid> {
        let at = self.i;
        if self.bump("-") {
            if let Some(n) = self.number()? {
                return Ok(Some(n));
            }
            self.i = at;
        }
        self.separator();
        if !["post", "rev", "r"]
            .iter()
            .any(|spelling| self.bump(spelling))
        {
            self.i = at;
            return Ok(None);
        }
        self.separator();
        Ok(Some(self.number()?.unwrap_or(0)))
    }

    fn dev(&mut self) -> Result<Option<u64>, Invalid> {
        let at = self.i;
        self.separator();
        if !self.bump("dev") {
            self.i = at;
            return Ok(None);
        }
        self.separator();
        Ok(Some(self.number()?.unwrap_or(0)))
    }

    fn local(&mut self) -> Result<Vec<String>, Invalid> {
        let mut segments = Vec::new();
        if !self.bump("+") {
            return Ok(segments);
        }
        loop {
            let start = self.i;
            while self.v.get(self.i).is_some_and(u8::is_ascii_alphanumeric) {
                self.i += 1;
            }
            let segment: String = self.v[start..self.i]
                .iter()
                .map(|&b| char::from(b.to_ascii_lowercase()))
                .collect();
            if segment.is_empty() {
                return Err(Invalid);
            }
            // A number that fits a u64 loses its leading zeros; anything else is kept, lowercased.
            segments.push(segment.parse::<u64>().map_or(segment, |n| n.to_string()));
            if !self.separator() {
                return Ok(segments);
            }
        }
    }

    /// The digits here as a number: `Ok(None)` if there are none, `Err` past `u64::MAX`.
    fn number(&mut self) -> Result<Option<u64>, Invalid> {
        let start = self.i;
        while self.v.get(self.i).is_some_and(u8::is_ascii_digit) {
            self.i += 1;
        }
        let digits = &self.v[start..self.i];
        if digits.is_empty() {
            return Ok(None);
        }
        digits
            .iter()
            .try_fold(0u64, |n, &d| {
                n.checked_mul(10)?.checked_add(u64::from(d - b'0'))
            })
            .map(Some)
            .ok_or(Invalid)
    }

    /// Consumes `s` (ASCII case-insensitively) if the input continues with it.
    fn bump(&mut self, s: &str) -> bool {
        let found = self
            .v
            .get(self.i..self.i + s.len())
            .is_some_and(|w| w.eq_ignore_ascii_case(s.as_bytes()));
        if found {
            self.i += s.len();
        }
        found
    }

    /// Consumes one `.`, `-` or `_`, if there is one.
    fn separator(&mut self) -> bool {
        let found = matches!(self.v.get(self.i), Some(b'.' | b'-' | b'_'));
        if found {
            self.i += 1;
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::pep440;

    #[test]
    fn release_only() {
        assert_eq!(pep440("1.0.0").as_deref(), Some("1.0.0"));
        assert_eq!(pep440("0.10.255").as_deref(), Some("0.10.255"));
        assert_eq!(
            pep440("18446744073709551615.0.0").as_deref(),
            Some("18446744073709551615.0.0")
        );
    }

    #[test]
    fn pre_post_dev() {
        for (cargo, python) in [
            ("1.0.0-dev", "1.0.0.dev0"),
            ("1.0.0-dev.1", "1.0.0.dev1"),
            ("1.0.0-dev1", "1.0.0.dev1"),
            ("1.0.0-DEV", "1.0.0.dev0"),
            ("1.0.0-alpha", "1.0.0a0"),
            ("1.0.0-alpha.1", "1.0.0a1"),
            ("1.0.0-alpha1", "1.0.0a1"),
            ("1.0.0-alpha-1", "1.0.0a1"),
            ("1.0.0-a.2", "1.0.0a2"),
            ("1.0.0-beta.2", "1.0.0b2"),
            ("1.0.0-b3", "1.0.0b3"),
            ("1.0.0-rc.1", "1.0.0rc1"),
            ("1.0.0-RC.1", "1.0.0rc1"),
            ("1.0.0-c.1", "1.0.0rc1"),
            ("1.0.0-pre.1", "1.0.0rc1"),
            ("1.0.0-preview.1", "1.0.0rc1"),
            ("1.0.0-rc01", "1.0.0rc1"),
            ("1.0.0-1", "1.0.0.post1"),
            ("1.0.0-post.2", "1.0.0.post2"),
            ("1.0.0-post", "1.0.0.post0"),
            ("1.0.0-rev.3", "1.0.0.post3"),
            ("1.0.0-r.4", "1.0.0.post4"),
            ("1.0.0-alpha.1.dev.2", "1.0.0a1.dev2"),
            ("1.0.0-rc.1.post.2.dev.3", "1.0.0rc1.post2.dev3"),
            ("1.0.0-beta-post-dev", "1.0.0b0.post0.dev0"),
        ] {
            assert_eq!(pep440(cargo).as_deref(), Some(python), "{cargo}");
        }
    }

    #[test]
    fn build_metadata_is_the_local_version() {
        for (cargo, python) in [
            ("1.0.0+abc", "1.0.0+abc"),
            ("1.0.0+Build-07", "1.0.0+build.7"),
            ("1.0.0+exp.sha.5114f85", "1.0.0+exp.sha.5114f85"),
            ("1.0.0-dev+20261106", "1.0.0.dev0+20261106"),
            (
                "1.0.0-rc.1+0.99999999999999999999",
                "1.0.0rc1+0.99999999999999999999",
            ),
        ] {
            assert_eq!(pep440(cargo).as_deref(), Some(python), "{cargo}");
        }
    }

    #[test]
    fn not_pep440() {
        for cargo in [
            "1.0.0-nightly",
            "1.0.0-x.7.z.92",
            "1.0.0-0.3.7",
            "1.0.0-alpha.beta",
            "1.0.0-dev.1.alpha",
            "1.0.0-alpha.99999999999999999999",
            "1.0.0+",
            "1.0.0+a..b",
            "",
        ] {
            assert_eq!(pep440(cargo), None, "{cargo}");
        }
    }

    #[test]
    fn this_crate() {
        assert!(pep440(env!("CARGO_PKG_VERSION")).is_some());
    }
}
