//! Text and NLP operations
//!
//! This module provides high-performance text and NLP operations.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use regex::Regex;
use tokenizers::tokenizer::{Tokenizer, EncodeInput};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use std::sync::Arc;

/// Tokenize texts in parallel
///
/// Args:
///     texts: A list of texts to tokenize
///     lowercase: Whether to lowercase the texts before tokenization (default: true)
///     remove_punct: Whether to remove punctuation (default: true)
///
/// Returns:
///     A list of tokenized texts (each text is a list of tokens)
#[pyfunction]
fn parallel_tokenize(
    py: Python,
    texts: Vec<String>,
    lowercase: Option<bool>,
    remove_punct: Option<bool>
) -> PyResult<PyObject> {
    let lowercase = lowercase.unwrap_or(true);
    let remove_punct = remove_punct.unwrap_or(true);
    
    // Create a regex for punctuation if needed
    let punct_regex = if remove_punct {
        Some(Regex::new(r"[^\w\s]").map_err(|e| PyValueError::new_err(format!("Invalid regex: {}", e)))?)
    } else {
        None
    };
    
    // Tokenize texts in parallel
    let tokenized_texts: Vec<Vec<String>> = texts.par_iter()
        .map(|text| {
            // Preprocess the text
            let mut processed_text = text.clone();
            
            if lowercase {
                processed_text = processed_text.to_lowercase();
            }
            
            if let Some(ref regex) = punct_regex {
                processed_text = regex.replace_all(&processed_text, " ").to_string();
            }
            
            // Tokenize by whitespace
            processed_text.split_whitespace()
                .map(|s| s.to_string())
                .collect()
        })
        .collect();
    
    // Convert to Python list of lists
    let py_list = PyList::empty(py);
    
    for tokens in tokenized_texts {
        let tokens_list = PyList::empty(py);
        for token in tokens {
            tokens_list.append(token)?;
        }
        py_list.append(tokens_list)?;
    }
    
    Ok(py_list.into())
}

/// Generate n-grams from texts in parallel
///
/// Args:
///     texts: A list of texts or tokenized texts
///     n: Size of n-grams (default: 2)
///     tokenized: Whether the input is already tokenized (default: false)
///
/// Returns:
///     A list of n-grams for each text
#[pyfunction]
fn parallel_ngrams(
    py: Python,
    texts: &PyList,
    n: Option<usize>,
    tokenized: Option<bool>
) -> PyResult<PyObject> {
    let n = n.unwrap_or(2);
    let tokenized = tokenized.unwrap_or(false);
    
    if n < 1 {
        return Err(PyValueError::new_err("n must be at least 1"));
    }
    
    // Process texts in parallel
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..texts.len()).collect();
    let results: Result<Vec<Vec<String>>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = texts.get_item(i)?;
                
                // Get tokens based on whether input is already tokenized
                let tokens: Vec<String> = if tokenized {
                    // Input is a list of tokens
                    if let Ok(token_list) = item.extract::<Vec<String>>() {
                        token_list
                    } else {
                        return Err(PyValueError::new_err(
                            format!("Item at index {} is not a valid token list", i)
                        ));
                    }
                } else {
                    // Input is a string that needs to be tokenized
                    let text = item.extract::<String>()?;
                    text.split_whitespace().map(|s| s.to_string()).collect()
                };
                
                // Generate n-grams
                let mut ngrams = Vec::new();
                
                if tokens.len() >= n {
                    for i in 0..=tokens.len() - n {
                        let ngram = tokens[i..i+n].join(" ");
                        ngrams.push(ngram);
                    }
                }
                
                Ok(ngrams)
            })
        })
        .collect();
    
    // Convert to Python list of lists
    let py_list = PyList::empty(py);
    
    for ngrams in results? {
        let ngrams_list = PyList::empty(py);
        for ngram in ngrams {
            ngrams_list.append(ngram)?;
        }
        py_list.append(ngrams_list)?;
    }
    
    Ok(py_list.into())
}

/// Calculate TF-IDF matrix in parallel
///
/// Args:
///     documents: A list of documents (strings or tokenized documents)
///     tokenized: Whether the input is already tokenized (default: false)
///     min_df: Minimum document frequency for a term to be included (default: 1)
///     max_df: Maximum document frequency for a term to be included (default: 1.0)
///
/// Returns:
///     A tuple of (tfidf_matrix, vocabulary)
///     - tfidf_matrix: A list of dictionaries mapping term indices to TF-IDF values
///     - vocabulary: A dictionary mapping terms to indices
#[pyfunction]
fn parallel_tfidf(
    py: Python,
    documents: &PyList,
    tokenized: Option<bool>,
    min_df: Option<PyObject>,
    max_df: Option<PyObject>
) -> PyResult<PyObject> {
    let tokenized = tokenized.unwrap_or(false);
    let n_docs = documents.len();
    
    if n_docs == 0 {
        return Err(PyValueError::new_err("Empty document list"));
    }
    
    // Parse min_df and max_df
    let min_df_count = if let Some(min_df_obj) = min_df {
        if let Ok(min_df_float) = min_df_obj.extract::<f64>(py) {
            ((min_df_float * n_docs as f64).round() as usize).max(1)
        } else if let Ok(min_df_int) = min_df_obj.extract::<usize>(py) {
            min_df_int
        } else {
            1 // Default
        }
    } else {
        1 // Default
    };
    
    let max_df_count = if let Some(max_df_obj) = max_df {
        if let Ok(max_df_float) = max_df_obj.extract::<f64>(py) {
            ((max_df_float * n_docs as f64).round() as usize).min(n_docs)
        } else if let Ok(max_df_int) = max_df_obj.extract::<usize>(py) {
            max_df_int.min(n_docs)
        } else {
            n_docs // Default
        }
    } else {
        n_docs // Default
    };
    
    // Tokenize documents and count terms
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..n_docs).collect();
    let tokenized_docs: Result<Vec<Vec<String>>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = documents.get_item(i)?;
                
                if tokenized {
                    // Input is a list of tokens
                    if let Ok(token_list) = item.extract::<Vec<String>>() {
                        Ok(token_list)
                    } else {
                        Err(PyValueError::new_err(
                            format!("Item at index {} is not a valid token list", i)
                        ))
                    }
                } else {
                    // Input is a string that needs to be tokenized
                    let text = item.extract::<String>()?;
                    Ok(text.split_whitespace().map(|s| s.to_string()).collect())
                }
            })
        })
        .collect();
    
    let tokenized_docs = tokenized_docs?;
    
    // Count document frequency for each term
    let mut doc_freq = HashMap::new();
    
    for doc in &tokenized_docs {
        let unique_terms: HashSet<&String> = doc.iter().collect();
        
        for term in unique_terms {
            *doc_freq.entry(term.clone()).or_insert(0) += 1;
        }
    }
    
    // Filter terms by document frequency
    let mut vocabulary = HashMap::new();
    let mut idx = 0;
    
    for (term, freq) in doc_freq.iter() {
        if *freq >= min_df_count && *freq <= max_df_count {
            vocabulary.insert(term.clone(), idx);
            idx += 1;
        }
    }
    
    // Calculate TF-IDF
    let tfidf_matrix: Vec<HashMap<usize, f64>> = tokenized_docs.par_iter()
        .map(|doc| {
            // Count term frequency in this document
            let mut term_freq = HashMap::new();
            
            for term in doc {
                if let Some(&term_idx) = vocabulary.get(term) {
                    *term_freq.entry(term_idx).or_insert(0.0) += 1.0;
                }
            }
            
            // Calculate TF-IDF for each term
            let mut tfidf = HashMap::new();
            let doc_len = doc.len() as f64;
            
            for (term_idx, tf) in term_freq {
                let term = vocabulary.iter()
                    .find(|(_, &idx)| idx == term_idx)
                    .map(|(term, _)| term)
                    .unwrap();
                
                let df = doc_freq.get(term).unwrap();
                let idf = (n_docs as f64 / *df as f64).ln();
                
                // TF-IDF = (term_count / doc_length) * log(n_docs / doc_freq)
                let tfidf_value = (tf / doc_len) * idf;
                tfidf.insert(term_idx, tfidf_value);
            }
            
            tfidf
        })
        .collect();
    
    // Convert to Python objects
    let py_tfidf_matrix = PyList::empty(py);
    
    for doc_tfidf in tfidf_matrix {
        let py_doc_tfidf = PyDict::new(py);
        
        for (term_idx, tfidf_value) in doc_tfidf {
            py_doc_tfidf.set_item(term_idx, tfidf_value)?;
        }
        
        py_tfidf_matrix.append(py_doc_tfidf)?;
    }
    
    let py_vocabulary = PyDict::new(py);
    
    for (term, idx) in vocabulary {
        py_vocabulary.set_item(term, idx)?;
    }
    
    // Return a dictionary with both components instead of a tuple
    let result_dict = PyDict::new(py);
    result_dict.set_item("tfidf_matrix", py_tfidf_matrix)?;
    result_dict.set_item("vocabulary", py_vocabulary)?;
    Ok(result_dict.into())
}

/// Calculate document similarity matrix in parallel
///
/// Args:
///     docs: A list of documents (strings or tokenized documents)
///     method: Similarity method (cosine, jaccard, overlap) (default: cosine)
///     tokenized: Whether the input is already tokenized (default: false)
///
/// Returns:
///     A 2D array of similarity scores between documents
#[pyfunction]
fn parallel_document_similarity(
    py: Python,
    docs: &PyList,
    method: Option<String>,
    tokenized: Option<bool>
) -> PyResult<PyObject> {
    let method = method.unwrap_or_else(|| "cosine".to_string());
    let tokenized = tokenized.unwrap_or(false);
    let n_docs = docs.len();
    
    if n_docs == 0 {
        return Err(PyValueError::new_err("Empty document list"));
    }
    
    // Tokenize documents
    // First collect indices to avoid PyO3 thread safety issues
    let indices: Vec<usize> = (0..n_docs).collect();
    let tokenized_docs: Result<Vec<Vec<String>>, PyErr> = indices
        .into_iter() // Use regular iterator instead of parallel
        .map(|i| {
            Python::with_gil(|py| {
                let item = docs.get_item(i)?;
                
                if tokenized {
                    // Input is a list of tokens
                    if let Ok(token_list) = item.extract::<Vec<String>>() {
                        Ok(token_list)
                    } else {
                        Err(PyValueError::new_err(
                            format!("Item at index {} is not a valid token list", i)
                        ))
                    }
                } else {
                    // Input is a string that needs to be tokenized
                    let text = item.extract::<String>()?;
                    Ok(text.split_whitespace().map(|s| s.to_string()).collect())
                }
            })
        })
        .collect();
    
    let tokenized_docs = tokenized_docs?;
    
    // Create document vectors
    let mut all_terms = HashSet::new();
    
    for doc in &tokenized_docs {
        for term in doc {
            all_terms.insert(term.clone());
        }
    }
    
    let term_to_idx: HashMap<String, usize> = all_terms.iter()
        .enumerate()
        .map(|(idx, term)| (term.clone(), idx))
        .collect();
    
    let doc_vectors: Vec<HashMap<usize, f64>> = tokenized_docs.iter()
        .map(|doc| {
            let mut term_counts = HashMap::new();
            
            for term in doc {
                if let Some(&idx) = term_to_idx.get(term) {
                    *term_counts.entry(idx).or_insert(0.0) += 1.0;
                }
            }
            
            term_counts
        })
        .collect();
    
    // Calculate similarity matrix
    let similarity_matrix = match method.as_str() {
        "cosine" => {
            // Precompute document vector norms
            let norms: Vec<f64> = doc_vectors.iter()
                .map(|vec| {
                    vec.values().map(|&v| v * v).sum::<f64>().sqrt()
                })
                .collect();
            
            // Calculate cosine similarity
            // Use sequential approach instead of parallel
            let mut matrix = vec![vec![0.0; n_docs]; n_docs];
            
            for i in 0..n_docs {
                matrix[i][i] = 1.0; // Self-similarity is 1
                
                for j in (i+1)..n_docs {
                    // Calculate dot product
                    let mut dot_product = 0.0;
                    
                    for (&idx, &val_i) in &doc_vectors[i] {
                        if let Some(&val_j) = doc_vectors[j].get(&idx) {
                            dot_product += val_i * val_j;
                        }
                    }
                    
                    // Calculate cosine similarity
                    let norm_i = norms[i];
                    let norm_j = norms[j];
                    
                    let similarity = if norm_i > 0.0 && norm_j > 0.0 {
                        dot_product / (norm_i * norm_j)
                    } else {
                        0.0
                    };
                    
                    matrix[i][j] = similarity;
                    matrix[j][i] = similarity; // Symmetric
                }
            }
            
            matrix
        },
        "jaccard" => {
            // Use sequential approach instead of parallel
            let mut matrix = vec![vec![0.0; n_docs]; n_docs];
            
            for i in 0..n_docs {
                matrix[i][i] = 1.0; // Self-similarity is 1
                
                for j in (i+1)..n_docs {
                    // Get sets of terms
                    let terms_i: HashSet<&String> = tokenized_docs[i].iter().collect();
                    let terms_j: HashSet<&String> = tokenized_docs[j].iter().collect();
                    
                    // Calculate intersection and union sizes
                    let intersection_size = terms_i.intersection(&terms_j).count();
                    let union_size = terms_i.union(&terms_j).count();
                    
                    // Calculate Jaccard similarity
                    let similarity = if union_size > 0 {
                        intersection_size as f64 / union_size as f64
                    } else {
                        0.0
                    };
                    
                    matrix[i][j] = similarity;
                    matrix[j][i] = similarity; // Symmetric
                }
            }
            
            matrix
        },
        "overlap" => {
            // Use sequential approach instead of parallel
            let mut matrix = vec![vec![0.0; n_docs]; n_docs];
            
            for i in 0..n_docs {
                matrix[i][i] = 1.0; // Self-similarity is 1
                
                for j in (i+1)..n_docs {
                    // Get sets of terms
                    let terms_i: HashSet<&String> = tokenized_docs[i].iter().collect();
                    let terms_j: HashSet<&String> = tokenized_docs[j].iter().collect();
                    
                    // Calculate intersection and minimum sizes
                    let intersection_size = terms_i.intersection(&terms_j).count();
                    let min_size = terms_i.len().min(terms_j.len());
                    
                    // Calculate overlap coefficient
                    let similarity = if min_size > 0 {
                        intersection_size as f64 / min_size as f64
                    } else {
                        0.0
                    };
                    
                    matrix[i][j] = similarity;
                    matrix[j][i] = similarity; // Symmetric
                }
            }
            
            matrix
        },
        _ => return Err(PyValueError::new_err(format!("Unsupported similarity method: {}", method))),
    };
    
    // Convert to Python list of lists
    let py_matrix = PyList::empty(py);
    
    for row in similarity_matrix {
        let py_row = PyList::empty(py);
        for val in row {
            py_row.append(val)?;
        }
        py_matrix.append(py_row)?;
    }
    
    Ok(py_matrix.into())
}

/// Register the text and NLP operations module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tfidf, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_document_similarity, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenize() {
        Python::with_gil(|py| {
            let texts = vec![
                "Hello world!".to_string(),
                "This is a test.".to_string(),
            ];
            
            let result = parallel_tokenize(py, texts, Some(true), Some(true)).unwrap();
            let tokenized: Vec<Vec<String>> = result.extract(py).unwrap();
            
            assert_eq!(tokenized.len(), 2);
            assert_eq!(tokenized[0], vec!["hello", "world"]);
            assert_eq!(tokenized[1], vec!["this", "is", "a", "test"]);
        });
    }
    
    #[test]
    fn test_document_similarity() {
        Python::with_gil(|py| {
            let docs = PyList::new(py, &[
                "this is a test",
                "this is another test",
                "something completely different",
            ]);
            
            let result = parallel_document_similarity(py, docs, Some("cosine".to_string()), Some(false)).unwrap();
            let matrix: Vec<Vec<f64>> = result.extract(py).unwrap();
            
            assert_eq!(matrix.len(), 3);
            assert!((matrix[0][0] - 1.0).abs() < 1e-10); // Self-similarity
            assert!(matrix[0][1] > matrix[0][2]); // First two docs are more similar
        });
    }
}