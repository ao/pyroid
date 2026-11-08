// Derived from sudachi.rs 0.7.0 (Apache-2.0), © Works Applications Co., Ltd.;
// modified.
//! `NumericParser` and `StringNumber`, which `JoinNumericPlugin` uses to
//! read Arabic and kanji numerals.

/// `numeric_parser::Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumError {
    None,
    Point,
    Comma,
}

#[derive(Debug, Default)]
struct StringNumber {
    significand: String,
    scale: usize,
    point: i32,
    is_all_zero: bool,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)] // as sudachi.rs: lengths are far below i32::MAX
impl StringNumber {
    fn new() -> Self {
        Self {
            significand: String::new(),
            scale: 0,
            point: -1,
            is_all_zero: true,
        }
    }

    fn clear(&mut self) {
        self.significand.clear();
        self.scale = 0;
        self.point = -1;
        self.is_all_zero = true;
    }

    fn append(&mut self, i: i32) {
        if i != 0 {
            self.is_all_zero = false;
        }
        self.significand += &i.to_string();
    }

    fn shift_scale(&mut self, i: i32) {
        if self.is_zero() {
            self.significand.push('1');
        }
        self.scale = (self.scale as i32 + i) as usize;
    }

    fn add(&mut self, number: &mut StringNumber) -> bool {
        if number.is_zero() {
            return true;
        }
        if self.is_zero() {
            self.significand.push_str(&number.significand);
            self.scale = number.scale;
            self.point = number.point;
            return true;
        }
        self.normalize_scale();
        let length = number.int_length();
        if self.scale >= length {
            self.fill_zero(self.scale - length);
            if number.point >= 0 {
                self.point = self.significand.len() as i32 + number.point;
            }
            self.significand.push_str(&number.significand);
            self.scale = number.scale;
            return true;
        }
        false
    }

    fn set_point(&mut self) -> bool {
        if self.scale == 0 && self.point < 0 {
            self.point = self.significand.len() as i32;
            return true;
        }
        false
    }

    fn int_length(&mut self) -> usize {
        self.normalize_scale();
        if self.point >= 0 {
            return self.point as usize;
        }
        self.significand.len() + self.scale
    }

    fn is_zero(&self) -> bool {
        self.significand.is_empty()
    }

    fn get_normalized(&mut self) -> String {
        if self.is_zero() {
            return "0".to_owned();
        }
        self.normalize_scale();
        if self.scale > 0 {
            self.fill_zero(self.scale);
        } else if self.point >= 0 {
            self.significand.insert(self.point as usize, '.');
            if self.point == 0 {
                self.significand.insert(0, '0');
            }
            let n_last_zero = self
                .significand
                .chars()
                .rev()
                .take_while(|c| *c == '0')
                .count();
            self.significand
                .truncate(self.significand.len() - n_last_zero);
            if self.significand.ends_with('.') {
                self.significand.truncate(self.significand.len() - 1);
            }
        }
        self.significand.clone()
    }

    fn normalize_scale(&mut self) {
        if self.point >= 0 {
            let n_scale = self.significand.len() as i32 - self.point;
            if n_scale > self.scale as i32 {
                self.point += self.scale as i32;
                self.scale = 0;
            } else {
                self.scale -= n_scale as usize;
                self.point = -1;
            }
        }
    }

    fn fill_zero(&mut self, length: usize) {
        self.significand.extend(std::iter::repeat_n('0', length));
    }
}

/// `NumericParser`.
#[derive(Debug)]
pub(crate) struct NumericParser {
    digit_length: usize,
    is_first_digit: bool,
    has_comma: bool,
    has_hanging_point: bool,
    pub error_state: NumError,
    total: StringNumber,
    subtotal: StringNumber,
    tmp: StringNumber,
}

fn char_to_num(c: char) -> Option<i32> {
    Some(match c {
        '〇' | '0' => 0,
        '一' | '1' => 1,
        '二' | '2' => 2,
        '三' | '3' => 3,
        '四' | '4' => 4,
        '五' | '5' => 5,
        '六' | '6' => 6,
        '七' | '7' => 7,
        '八' | '8' => 8,
        '九' | '9' => 9,
        '十' => -1,
        '百' => -2,
        '千' => -3,
        '万' => -4,
        '億' => -8,
        '兆' => -12,
        _ => return None,
    })
}

impl NumericParser {
    pub fn new() -> Self {
        Self {
            digit_length: 0,
            is_first_digit: true,
            has_comma: false,
            has_hanging_point: false,
            error_state: NumError::None,
            total: StringNumber::new(),
            subtotal: StringNumber::new(),
            tmp: StringNumber::new(),
        }
    }

    pub fn clear(&mut self) {
        self.digit_length = 0;
        self.is_first_digit = true;
        self.has_comma = false;
        self.has_hanging_point = false;
        self.error_state = NumError::None;
        self.total.clear();
        self.subtotal.clear();
        self.tmp.clear();
    }

    pub fn append(&mut self, c: char) -> bool {
        if c == '.' {
            self.has_hanging_point = true;
            if self.is_first_digit {
                self.error_state = NumError::Point;
                return false;
            }
            if self.has_comma && !self.check_comma() {
                self.error_state = NumError::Comma;
                return false;
            }
            if !self.tmp.set_point() {
                self.error_state = NumError::Point;
                return false;
            }
            self.has_comma = false;
            return true;
        }
        if c == ',' {
            if !self.check_comma() {
                self.error_state = NumError::Comma;
                return false;
            }
            self.has_comma = true;
            self.digit_length = 0;
            return true;
        }
        let Some(n) = char_to_num(c) else {
            return false;
        };
        if (-3..0).contains(&n) {
            self.tmp.shift_scale(-n);
            if !self.subtotal.add(&mut self.tmp) {
                return false;
            }
            self.tmp.clear();
            self.is_first_digit = true;
            self.digit_length = 0;
            self.has_comma = false;
        } else if n < -3 {
            if !self.subtotal.add(&mut self.tmp) || self.subtotal.is_zero() {
                return false;
            }
            self.subtotal.shift_scale(-n);
            if !self.total.add(&mut self.subtotal) {
                return false;
            }
            self.subtotal.clear();
            self.tmp.clear();
            self.is_first_digit = true;
            self.digit_length = 0;
            self.has_comma = false;
        } else {
            self.tmp.append(n);
            self.is_first_digit = false;
            self.digit_length += 1;
            self.has_hanging_point = false;
        }
        true
    }

    pub fn done(&mut self) -> bool {
        let ret = self.subtotal.add(&mut self.tmp) && self.total.add(&mut self.subtotal);
        if self.has_hanging_point {
            self.error_state = NumError::Point;
            return false;
        }
        if self.has_comma && self.digit_length != 3 {
            self.error_state = NumError::Comma;
            return false;
        }
        ret
    }

    pub fn get_normalized(&mut self) -> String {
        self.total.get_normalized()
    }

    fn check_comma(&self) -> bool {
        if self.is_first_digit {
            return false;
        }
        if !self.has_comma {
            return self.digit_length <= 3 && !self.tmp.is_zero() && !self.tmp.is_all_zero;
        }
        self.digit_length == 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Option<String> {
        let mut p = NumericParser::new();
        for c in s.chars() {
            if !p.append(c) {
                return None;
            }
        }
        p.done().then(|| p.get_normalized())
    }

    #[test]
    fn sudachi_examples() {
        // The cases of sudachi.rs 0.7.0's numeric_parser tests.
        let ok = [
            ("1000", "1000"),
            ("001000", "001000"),
            ("〇一〇〇〇", "01000"),
            ("00.1000", "00.1"),
            ("000", "000"),
            ("二十七", "27"),
            ("千三百二十七", "1327"),
            ("千十七", "1017"),
            ("千三百二十七.〇五", "1327.05"),
            ("1万", "10000"),
            ("千三百二十七万", "13270000"),
            ("千三百二十七万一四", "13270014"),
            ("千三百二十七万一四.〇五", "13270014.05"),
            ("三兆2千億千三百二十七万一四.〇五", "3200013270014.05"),
            ("1.5千", "1500"),
            ("1.5百万", "1500000"),
            ("1.5百万1.5千20", "1501520"),
            ("200000000000000000000万", "2000000000000000000000000"),
            ("2,000,000", "2000000"),
            ("259万2,300", "2592300"),
        ];
        for (s, want) in ok {
            assert_eq!(parse(s).as_deref(), Some(want), "{s}");
        }
        for s in ["三百二十百", "億万", "1.5千5百", "1.5千500", "@@@"] {
            assert_eq!(parse(s), None, "{s}");
        }
        for s in ["200,00,000", "2,4", "000,000", ",000", "256,55.1"] {
            let mut p = NumericParser::new();
            let ok = s.chars().all(|c| p.append(c)) && p.done();
            assert!(!ok, "{s}");
            assert_eq!(p.error_state, NumError::Comma, "{s}");
        }
    }
}
