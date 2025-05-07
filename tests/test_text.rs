//! Integration tests for the text module

use pyroid::text;

#[test]
fn test_reverse() {
    // Test with a simple string
    let result = text::reverse("hello");
    assert_eq!(result, "olleh");
    
    // Test with an empty string
    let result = text::reverse("");
    assert_eq!(result, "");
    
    // Test with a string containing spaces
    let result = text::reverse("hello world");
    assert_eq!(result, "dlrow olleh");
    
    // Test with Unicode characters
    let result = text::reverse("こんにちは");
    assert_eq!(result, "はちにんこ");
}

#[test]
fn test_base64() {
    // Test encoding
    let result = text::base64_encode("hello");
    assert_eq!(result, "aGVsbG8=");
    
    // Test decoding
    let result = text::base64_decode("aGVsbG8=");
    assert_eq!(result, "hello");
    
    // Test round trip
    let original = "Hello, world! 123";
    let encoded = text::base64_encode(original);
    let decoded = text::base64_decode(&encoded);
    assert_eq!(decoded, original);
    
    // Test with empty string
    let result = text::base64_encode("");
    assert_eq!(result, "");
    
    let result = text::base64_decode("");
    assert_eq!(result, "");
}

#[test]
fn test_split() {
    // Test with a simple string
    let result = text::split("a,b,c", ",");
    assert_eq!(result, vec!["a", "b", "c"]);
    
    // Test with an empty string
    let result = text::split("", ",");
    assert_eq!(result, vec![""]);
    
    // Test with a string that doesn't contain the delimiter
    let result = text::split("abc", ",");
    assert_eq!(result, vec!["abc"]);
    
    // Test with multiple consecutive delimiters
    let result = text::split("a,,c", ",");
    assert_eq!(result, vec!["a", "", "c"]);
    
    // Test with delimiter at the beginning and end
    let result = text::split(",a,b,c,", ",");
    assert_eq!(result, vec!["", "a", "b", "c", ""]);
}

#[test]
fn test_join() {
    // Test with a list of strings
    let result = text::join(&["a", "b", "c"], ",");
    assert_eq!(result, "a,b,c");
    
    // Test with an empty list
    let result = text::join(&[], ",");
    assert_eq!(result, "");
    
    // Test with a list containing empty strings
    let result = text::join(&["", "", ""], ",");
    assert_eq!(result, ",,");
    
    // Test with a different delimiter
    let result = text::join(&["a", "b", "c"], " - ");
    assert_eq!(result, "a - b - c");
}

#[test]
fn test_replace() {
    // Test with a simple string
    let result = text::replace("hello world", "world", "universe");
    assert_eq!(result, "hello universe");
    
    // Test with a string that doesn't contain the substring
    let result = text::replace("hello world", "universe", "galaxy");
    assert_eq!(result, "hello world");
    
    // Test with an empty string
    let result = text::replace("", "world", "universe");
    assert_eq!(result, "");
    
    // Test with empty replacement
    let result = text::replace("hello world", "world", "");
    assert_eq!(result, "hello ");
    
    // Test with empty search string
    let result = text::replace("hello world", "", "x");
    assert_eq!(result, "hello world");
    
    // Test with multiple occurrences
    let result = text::replace("hello hello hello", "hello", "hi");
    assert_eq!(result, "hi hi hi");
}

#[test]
fn test_regex_replace() {
    // Test with a simple pattern
    let result = text::regex_replace("hello 123 world", r"\d+", "456");
    assert_eq!(result, "hello 456 world");
    
    // Test with a pattern that doesn't match
    let result = text::regex_replace("hello world", r"\d+", "456");
    assert_eq!(result, "hello world");
    
    // Test with a more complex pattern
    let result = text::regex_replace("hello world", r"(hello) (world)", "$2 $1");
    assert_eq!(result, "world hello");
    
    // Test with multiple matches
    let result = text::regex_replace("123 456 789", r"\d+", "0");
    assert_eq!(result, "0 0 0");
    
    // Test with empty string
    let result = text::regex_replace("", r"\d+", "0");
    assert_eq!(result, "");
}

#[test]
fn test_to_uppercase() {
    // Test with a simple string
    let result = text::to_uppercase("hello");
    assert_eq!(result, "HELLO");
    
    // Test with an empty string
    let result = text::to_uppercase("");
    assert_eq!(result, "");
    
    // Test with a mixed case string
    let result = text::to_uppercase("Hello World");
    assert_eq!(result, "HELLO WORLD");
    
    // Test with non-alphabetic characters
    let result = text::to_uppercase("hello123!@#");
    assert_eq!(result, "HELLO123!@#");
}

#[test]
fn test_to_lowercase() {
    // Test with a simple string
    let result = text::to_lowercase("HELLO");
    assert_eq!(result, "hello");
    
    // Test with an empty string
    let result = text::to_lowercase("");
    assert_eq!(result, "");
    
    // Test with a mixed case string
    let result = text::to_lowercase("Hello World");
    assert_eq!(result, "hello world");
    
    // Test with non-alphabetic characters
    let result = text::to_lowercase("HELLO123!@#");
    assert_eq!(result, "hello123!@#");
}

#[test]
fn test_tokenize() {
    // Test with a simple string
    let result = text::tokenize("hello world");
    assert_eq!(result, vec!["hello", "world"]);
    
    // Test with lowercase=false
    let result = text::tokenize_with_options("Hello World", false, true);
    assert_eq!(result, vec!["Hello", "World"]);
    
    // Test with remove_punct=false
    let result = text::tokenize_with_options("hello, world!", true, false);
    assert_eq!(result, vec!["hello,", "world!"]);
    
    // Test with an empty string
    let result = text::tokenize("");
    assert_eq!(result, Vec::<String>::new());
}

#[test]
fn test_ngrams() {
    // Test with a string
    let result = text::ngrams("hello world", 2);
    assert_eq!(result, vec![vec!["hello", "world"]]);
    
    // Test with a list of tokens
    let result = text::ngrams_from_tokens(&["a", "b", "c", "d"], 2);
    assert_eq!(result, vec![vec!["a", "b"], vec!["b", "c"], vec!["c", "d"]]);
    
    // Test with n > len(tokens)
    let result = text::ngrams_from_tokens(&["a", "b"], 3);
    assert_eq!(result, Vec::<Vec<String>>::new());
    
    // Test with an empty list
    let result = text::ngrams_from_tokens(&[], 2);
    assert_eq!(result, Vec::<Vec<String>>::new());
}